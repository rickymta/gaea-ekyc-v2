from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class EKYCStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AssetType(str, Enum):
    ID_FRONT = "id_front"
    ID_BACK = "id_back"
    SELFIE = "selfie"


class FinalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"


# Request Schemas
class EKYCSessionCreate(BaseModel):
    user_id: str = Field(..., description="User identifier", example="user123")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "testuser"
            }
        }


class FileUploadResponse(BaseModel):
    asset_id: UUID
    message: str
    file_path: str
    
    class Config:
        schema_extra = {
            "example": {
                "asset_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "File uploaded successfully",
                "file_path": "ekyc-assets/sessions/session-id/id_front/file.jpg"
            }
        }


# Response Schemas
class EKYCAssetResponse(BaseModel):
    id: UUID
    session_id: UUID
    asset_type: AssetType
    file_path: str
    original_filename: Optional[str] = None
    file_size: Optional[float] = None
    mime_type: Optional[str] = None
    processed: bool
    processing_result: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ProcessingStages(BaseModel):
    id_document_uploaded: bool = False
    id_document_processed: bool = False
    selfie_uploaded: bool = False
    selfie_processed: bool = False
    face_match_completed: bool = False


class IDCardData(BaseModel):
    id_number: Optional[str] = Field(None, example="123456789012")
    full_name: Optional[str] = Field(None, example="NGUYEN VAN A")
    date_of_birth: Optional[str] = Field(None, example="01/01/1990")
    gender: Optional[str] = Field(None, example="Nam")
    nationality: Optional[str] = Field(None, example="Việt Nam")
    address: Optional[str] = Field(None, example="123 Đường ABC, Quận XYZ, TP. HCM")
    issue_date: Optional[str] = Field(None, example="01/01/2020")
    expiry_date: Optional[str] = Field(None, example="01/01/2030")
    place_of_origin: Optional[str] = Field(None, example="TP. Hồ Chí Minh")
    place_of_residence: Optional[str] = Field(None, example="TP. Hồ Chí Minh")
    confidence_scores: Optional[Dict[str, float]] = None


class EKYCSessionResponse(BaseModel):
    id: UUID
    user_id: str
    status: EKYCStatus
    id_card_data: Optional[IDCardData] = None
    face_match_score: Optional[float] = None
    liveness_score: Optional[float] = None
    final_decision: Optional[FinalDecision] = None
    error_message: Optional[str] = None
    processing_stages: Optional[ProcessingStages] = None
    created_at: datetime
    updated_at: datetime
    assets: List[EKYCAssetResponse] = []

    class Config:
        from_attributes = True


class EKYCSessionListResponse(BaseModel):
    sessions: List[EKYCSessionResponse]
    total: int
    page: int
    page_size: int


# Error Schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ValidationErrorResponse(BaseModel):
    error: str = "validation_error"
    message: str
    details: List[Dict[str, Any]]


# JWT Schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None


class User(BaseModel):
    user_id: str
    email: Optional[str] = None
    is_active: bool = True


# Task Schemas
class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Processing Result Schemas
class FaceDetectionResult(BaseModel):
    detected: bool
    confidence: float
    bounding_box: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None


class LivenessResult(BaseModel):
    is_live: bool
    score: float
    checks: Dict[str, Any]


class FaceMatchResult(BaseModel):
    match: bool
    similarity_score: float
    threshold_used: float


class OCRResult(BaseModel):
    extracted_data: IDCardData
    confidence_score: float
    processing_time: float


class ProcessingTaskResult(BaseModel):
    task_type: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
