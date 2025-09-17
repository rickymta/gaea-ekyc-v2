from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app.database import get_db
from app.dependencies import get_current_active_user
from app.schemas import (
    FileUploadResponse, EKYCAssetResponse, AssetType, 
    ErrorResponse, User, TaskStatus
)
from app.services.session_service import SessionService
from app.services.asset_service import AssetService
from app.tasks.processing_tasks import process_id_card_task, process_selfie_task
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["File Upload"])

# Allowed file types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/sessions/{session_id}/upload",
    response_model=FileUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        413: {"model": ErrorResponse, "description": "File too large"}
    },
    summary="Upload Asset File",
    description="""
    **Upload ảnh CMND hoặc selfie cho session EKYC**
    
    ### Supported Asset Types:
    - `id_front`: Ảnh mặt trước CMND/CCCD
    - `id_back`: Ảnh mặt sau CMND/CCCD  
    - `selfie`: Ảnh selfie để xác minh khuôn mặt
    
    ### File Requirements:
    - Format: JPEG, JPG, PNG
    - Max size: 10MB
    - Recommended: High resolution, well-lit photos
    """
)
async def upload_asset(
    session_id: UUID,
    asset_type: AssetType = Form(..., description="Type of asset being uploaded"),
    file: UploadFile = File(..., description="Image file to upload"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload file for EKYC session"""
    
    # Validate session exists and belongs to user
    session = SessionService.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EKYC session not found"
        )
    
    if session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check if asset type already exists
    if AssetService.asset_exists_for_session(db, session_id, asset_type):
        # Delete existing asset of this type
        existing_asset = AssetService.get_session_asset_by_type(db, session_id, asset_type)
        if existing_asset:
            AssetService.delete_asset(db, existing_asset.id)
            logger.info(f"Replaced existing {asset_type.value} for session {session_id}")
    
    try:
        # Upload file and create asset record
        asset = await AssetService.upload_file(db, session_id, asset_type, file)
        
        # Update session processing stage
        stage_key = f"{asset_type.value}_uploaded"
        SessionService.update_processing_stage(db, session_id, stage_key, True)
        
        # Trigger background processing
        if asset_type in [AssetType.ID_FRONT, AssetType.ID_BACK]:
            task = process_id_card_task.delay(str(asset.id))
            logger.info(f"Started ID card processing task {task.id} for asset {asset.id}")
        elif asset_type == AssetType.SELFIE:
            task = process_selfie_task.delay(str(asset.id))
            logger.info(f"Started selfie processing task {task.id} for asset {asset.id}")
        
        return {
            "asset_id": asset.id,
            "message": f"{asset_type.value} uploaded successfully",
            "file_path": asset.file_path
        }
        
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )


@router.get(
    "/sessions/{session_id}/assets",
    response_model=List[EKYCAssetResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="List Session Assets",
    description="Lấy danh sách tất cả assets của một session"
)
async def list_session_assets(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all assets for a session"""
    session = SessionService.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EKYC session not found"
        )
    
    if session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    assets = AssetService.get_session_assets(db, session_id)
    return assets


@router.get(
    "/assets/{asset_id}",
    response_model=EKYCAssetResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Asset not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Get Asset Details",
    description="Lấy thông tin chi tiết của một asset"
)
async def get_asset(
    asset_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get asset by ID"""
    asset = AssetService.get_asset(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    # Check if user owns the session this asset belongs to
    session = SessionService.get_session(db, asset.session_id)
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this asset"
        )
    
    return asset


@router.get(
    "/assets/{asset_id}/download",
    responses={
        200: {"description": "Presigned URL for file download"},
        404: {"model": ErrorResponse, "description": "Asset not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Download Asset File",
    description="Lấy presigned URL để download file asset"
)
async def download_asset(
    asset_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get presigned URL to download asset file"""
    asset = AssetService.get_asset(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    # Check if user owns the session this asset belongs to
    session = SessionService.get_session(db, asset.session_id)
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this asset"
        )
    
    download_url = AssetService.get_file_url(asset, expires_hours=1)
    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )
    
    return {"download_url": download_url, "expires_in": "1 hour"}


@router.delete(
    "/assets/{asset_id}",
    responses={
        200: {"description": "Asset deleted successfully"},
        404: {"model": ErrorResponse, "description": "Asset not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Delete Asset",
    description="Xóa asset và file liên quan"
)
async def delete_asset(
    asset_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete asset"""
    asset = AssetService.get_asset(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    # Check if user owns the session this asset belongs to
    session = SessionService.get_session(db, asset.session_id)
    if not session or session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this asset"
        )
    
    success = AssetService.delete_asset(db, asset_id)
    if success:
        return {"message": "Asset deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete asset"
        )
