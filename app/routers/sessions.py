from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from app.database import get_db
from app.dependencies import get_current_active_user
from app.schemas import (
    EKYCSessionCreate, EKYCSessionResponse, EKYCSessionListResponse,
    ErrorResponse, User
)
from app.services.session_service import SessionService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["EKYC Sessions"])


@router.post(
    "/sessions",
    response_model=EKYCSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Create EKYC Session",
    description="""
    **Tạo session EKYC mới**
    
    Mỗi session đại diện cho một quy trình xác minh danh tính hoàn chỉnh.
    Session sẽ tracking toàn bộ quá trình từ upload ảnh đến kết quả cuối cùng.
    """
)
async def create_ekyc_session(
    session_data: EKYCSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new EKYC session"""
    try:
        # Override user_id from token for security
        session_data.user_id = current_user.user_id
        
        session = SessionService.create_session(db, session_data)
        logger.info(f"Created EKYC session {session.id} for user {current_user.user_id}")
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to create EKYC session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create EKYC session"
        )


@router.get(
    "/sessions/{session_id}",
    response_model=EKYCSessionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Get EKYC Session",
    description="Lấy thông tin chi tiết của một session EKYC"
)
async def get_ekyc_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get EKYC session by ID"""
    session = SessionService.get_session(db, session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EKYC session not found"
        )
    
    # Check if user owns this session
    if session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    return session


@router.get(
    "/sessions",
    response_model=EKYCSessionListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="List User Sessions",
    description="Lấy danh sách tất cả sessions của user hiện tại với phân trang"
)
async def list_user_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all sessions for current user with pagination"""
    skip = (page - 1) * page_size
    
    sessions = SessionService.get_user_sessions(
        db, current_user.user_id, skip=skip, limit=page_size
    )
    total = SessionService.get_user_sessions_count(db, current_user.user_id)
    
    return {
        "sessions": sessions,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.delete(
    "/sessions/{session_id}",
    responses={
        200: {"description": "Session deleted successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Delete EKYC Session",
    description="Xóa session EKYC và tất cả dữ liệu liên quan"
)
async def delete_ekyc_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete EKYC session"""
    session = SessionService.get_session(db, session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EKYC session not found"
        )
    
    # Check if user owns this session
    if session.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    success = SessionService.delete_session(db, session_id)
    
    if success:
        logger.info(f"Session {session_id} deleted by user {current_user.user_id}")
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )
