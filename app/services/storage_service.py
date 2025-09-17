import asyncio
import aiofiles
from minio import Minio
from minio.error import S3Error
from typing import Optional, BinaryIO, Tuple, Dict, Any
import uuid
import os
from datetime import timedelta
import logging
from app.config import settings

# Logging setup
logger = logging.getLogger(__name__)


class MinIOStorage:
    def __init__(self):
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_name = settings.minio_bucket_name
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket already exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error creating bucket: {str(e)}")
            raise
    
    def upload_file(self, file_data: BinaryIO, object_name: str, content_type: str = None) -> str:
        """Upload a file to MinIO storage"""
        try:
            # Get file size
            file_data.seek(0, 2)  # Seek to end
            file_size = file_data.tell()
            file_data.seek(0)  # Reset to beginning
            
            # Upload file
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type
            )
            
            logger.info(f"Successfully uploaded {object_name} to {self.bucket_name}")
            return f"{self.bucket_name}/{object_name}"
            
        except S3Error as e:
            logger.error(f"Error uploading file {object_name}: {str(e)}")
            raise
    
    async def upload_file_async(self, file_path: str, object_name: str, content_type: str = None) -> str:
        """Upload a file asynchronously"""
        def _upload():
            with open(file_path, 'rb') as file_data:
                return self.upload_file(file_data, object_name, content_type)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _upload)
    
    def download_file(self, object_name: str, file_path: str) -> bool:
        """Download a file from MinIO storage"""
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=file_path
            )
            logger.info(f"Successfully downloaded {object_name} to {file_path}")
            return True
        except S3Error as e:
            logger.error(f"Error downloading file {object_name}: {str(e)}")
            return False
    
    async def download_file_async(self, object_name: str, file_path: str) -> bool:
        """Download a file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.download_file, object_name, file_path)
    
    def get_file_data(self, object_name: str) -> Optional[bytes]:
        """Get file data as bytes"""
        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Error getting file data {object_name}: {str(e)}")
            return None
    
    async def get_file_data_async(self, object_name: str) -> Optional[bytes]:
        """Get file data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_file_data, object_name)
    
    def delete_file(self, object_name: str) -> bool:
        """Delete a file from MinIO storage"""
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            logger.info(f"Successfully deleted {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting file {object_name}: {str(e)}")
            return False
    
    def generate_presigned_url(self, object_name: str, expires: timedelta = timedelta(hours=1)) -> Optional[str]:
        """Generate a presigned URL for file access"""
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {object_name}: {str(e)}")
            return None
    
    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in storage"""
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        try:
            stat = self.client.stat_object(self.bucket_name, object_name)
            return {
                'size': stat.size,
                'last_modified': stat.last_modified,
                'content_type': stat.content_type,
                'etag': stat.etag
            }
        except S3Error as e:
            logger.error(f"Error getting file info {object_name}: {str(e)}")
            return None


class FileManager:
    """Higher-level file management with session organization"""
    
    def __init__(self):
        self.storage = MinIOStorage()
    
    def generate_object_path(self, session_id: str, asset_type: str, filename: str) -> str:
        """Generate organized object path"""
        file_id = str(uuid.uuid4())
        extension = os.path.splitext(filename)[1]
        return f"sessions/{session_id}/{asset_type}/{file_id}{extension}"
    
    async def save_uploaded_file(
        self, 
        session_id: str, 
        asset_type: str, 
        file_data: BinaryIO, 
        filename: str, 
        content_type: str
    ) -> Tuple[str, str]:
        """Save uploaded file and return object path and file path"""
        object_name = self.generate_object_path(session_id, asset_type, filename)
        file_path = self.storage.upload_file(file_data, object_name, content_type)
        return object_name, file_path
    
    async def get_file_for_processing(self, object_name: str) -> Optional[str]:
        """Download file to temporary location for processing"""
        temp_path = f"/tmp/{uuid.uuid4()}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        success = await self.storage.download_file_async(object_name, temp_path)
        return temp_path if success else None
    
    async def cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temp file {file_path}: {str(e)}")
    
    def get_file_url(self, object_name: str, expires_hours: int = 1) -> Optional[str]:
        """Get presigned URL for file access"""
        return self.storage.generate_presigned_url(
            object_name, 
            expires=timedelta(hours=expires_hours)
        )


# Global instance
storage_manager = MinIOStorage()
file_manager = FileManager()
