import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://ekyc_user:ekyc_password@localhost:5432/ekyc_db"
    
    # JWT
    secret_key: str = "your-super-secret-jwt-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "ekyc-assets"
    minio_secure: bool = False
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Application
    app_name: str = "EKYC Service"
    version: str = "1.0.0"
    debug: bool = True
    
    # InsightFace
    insightface_model: str = "model-r100-ii"
    insightface_model_path: str = "./models"
    face_match_threshold: float = 0.6
    liveness_threshold: float = 0.5
    face_detection_size: tuple = (640, 640)
    face_embedding_size: int = 512
    
    # File Upload
    max_file_size_mb: int = 10
    allowed_image_extensions: List[str] = ["jpg", "jpeg", "png"]
    
    # Logging
    log_level: str = "INFO"
    
    # Webhooks
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
