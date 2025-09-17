from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from uuid import UUID
from fastapi import UploadFile
from app.models import EKYCAsset, EKYCSession
from app.schemas import AssetType
from app.services.storage_service import file_manager
import logging

logger = logging.getLogger(__name__)


class AssetService:
    @staticmethod
    def create_asset(
        db: Session, 
        session_id: UUID,
        asset_type: AssetType,
        file_path: str,
        original_filename: str,
        file_size: float,
        mime_type: str
    ) -> EKYCAsset:
        """Create a new asset record"""
        db_asset = EKYCAsset(
            session_id=session_id,
            asset_type=asset_type.value,
            file_path=file_path,
            original_filename=original_filename,
            file_size=file_size,
            mime_type=mime_type,
            processed=False
        )
        db.add(db_asset)
        db.commit()
        db.refresh(db_asset)
        logger.info(f"Created asset {db_asset.id} for session {session_id}")
        return db_asset

    @staticmethod
    def get_asset(db: Session, asset_id: UUID) -> Optional[EKYCAsset]:
        """Get asset by ID"""
        return db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()

    @staticmethod
    def get_session_assets(
        db: Session, 
        session_id: UUID, 
        asset_type: Optional[AssetType] = None
    ) -> List[EKYCAsset]:
        """Get all assets for a session, optionally filtered by type"""
        query = db.query(EKYCAsset).filter(EKYCAsset.session_id == session_id)
        
        if asset_type:
            query = query.filter(EKYCAsset.asset_type == asset_type.value)
        
        return query.order_by(desc(EKYCAsset.created_at)).all()

    @staticmethod
    def get_session_asset_by_type(
        db: Session, 
        session_id: UUID, 
        asset_type: AssetType
    ) -> Optional[EKYCAsset]:
        """Get the most recent asset of a specific type for a session"""
        return (
            db.query(EKYCAsset)
            .filter(
                EKYCAsset.session_id == session_id,
                EKYCAsset.asset_type == asset_type.value
            )
            .order_by(desc(EKYCAsset.created_at))
            .first()
        )

    @staticmethod
    async def upload_file(
        db: Session,
        session_id: UUID,
        asset_type: AssetType,
        file: UploadFile
    ) -> EKYCAsset:
        """Upload file and create asset record"""
        # Save file to storage
        object_name, file_path = await file_manager.save_uploaded_file(
            session_id=str(session_id),
            asset_type=asset_type.value,
            file_data=file.file,
            filename=file.filename,
            content_type=file.content_type
        )
        
        # Create asset record
        asset = AssetService.create_asset(
            db=db,
            session_id=session_id,
            asset_type=asset_type,
            file_path=object_name,  # Store object name for retrieval
            original_filename=file.filename,
            file_size=file.size if hasattr(file, 'size') else 0,
            mime_type=file.content_type
        )
        
        logger.info(f"Uploaded file {file.filename} as asset {asset.id}")
        return asset

    @staticmethod
    def update_processing_result(
        db: Session,
        asset_id: UUID,
        processing_result: Dict[str, Any],
        processed: bool = True
    ) -> Optional[EKYCAsset]:
        """Update asset processing result"""
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if asset:
            asset.processing_result = processing_result
            asset.processed = processed
            db.commit()
            db.refresh(asset)
            logger.info(f"Updated processing result for asset {asset_id}")
        return asset

    @staticmethod
    def delete_asset(db: Session, asset_id: UUID) -> bool:
        """Delete asset and its file"""
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if asset:
            # Delete file from storage
            file_manager.storage.delete_file(asset.file_path)
            
            # Delete database record
            db.delete(asset)
            db.commit()
            logger.info(f"Deleted asset {asset_id}")
            return True
        return False

    @staticmethod
    def get_file_url(asset: EKYCAsset, expires_hours: int = 1) -> Optional[str]:
        """Get presigned URL for asset file"""
        return file_manager.get_file_url(asset.file_path, expires_hours)

    @staticmethod
    def asset_exists_for_session(
        db: Session, 
        session_id: UUID, 
        asset_type: AssetType
    ) -> bool:
        """Check if asset type already exists for session"""
        asset = AssetService.get_session_asset_by_type(db, session_id, asset_type)
        return asset is not None

    @staticmethod
    def get_unprocessed_assets(db: Session, limit: int = 10) -> List[EKYCAsset]:
        """Get unprocessed assets for background processing"""
        return (
            db.query(EKYCAsset)
            .filter(EKYCAsset.processed == False)
            .order_by(EKYCAsset.created_at)
            .limit(limit)
            .all()
        )

    @staticmethod
    def mark_asset_processed(
        db: Session, 
        asset_id: UUID, 
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[EKYCAsset]:
        """Mark asset as processed with result"""
        asset = db.query(EKYCAsset).filter(EKYCAsset.id == asset_id).first()
        if asset:
            asset.processed = True
            if not success and error_message:
                asset.processing_result = {"error": error_message, "success": False}
            elif success:
                if asset.processing_result is None:
                    asset.processing_result = {}
                asset.processing_result["success"] = True
            
            db.commit()
            db.refresh(asset)
            logger.info(f"Marked asset {asset_id} as processed (success: {success})")
        return asset
