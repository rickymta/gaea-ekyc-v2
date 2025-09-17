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
    app_name: str = "Enhanced EKYC Service"
    version: str = "2.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # InsightFace
    insightface_model: str = "model-r100-ii"
    insightface_model_path: str = "./models"
    face_match_threshold: float = 0.6
    liveness_threshold: float = 0.5
    face_detection_size: tuple = (640, 640)
    face_embedding_size: int = 512
    
    # Enhanced EKYC Settings
    # Liveness Detection
    liveness_model_path: str = "./models/liveness"
    liveness_confidence_threshold: float = 0.7
    liveness_fallback_enabled: bool = True
    
    # OCR Settings
    paddleocr_use_gpu: bool = False
    paddleocr_lang: str = "vi"
    easyocr_gpu: bool = False
    easyocr_languages: List[str] = ["vi", "en"]
    tesseract_path: str = ""  # Auto-detect if empty
    
    # Image Quality Settings
    min_face_size: int = 80
    max_face_size: int = 400
    min_brightness: float = 80.0
    max_brightness: float = 220.0
    min_sharpness: float = 100.0
    blur_threshold: float = 100.0
    
    # Video Processing
    max_video_size_mb: int = 10
    video_frame_rate: int = 5  # frames per second to analyze
    min_video_duration: float = 2.0  # seconds
    max_video_duration: float = 5.0  # seconds
    
    # Database Settings for Enhanced EKYC
    enable_result_storage: bool = True
    result_retention_days: int = 30
    
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
