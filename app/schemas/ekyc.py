"""
EKYC Verification Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class EKYCVerificationRequest(BaseModel):
    """Schema for EKYC verification request"""
    user_data: Optional[Dict[str, Any]] = Field(None, description="Additional user information")
    session_id: Optional[str] = Field(None, description="Session identifier")
    verification_type: str = Field("complete", description="Type of verification: complete, face_only, liveness_only")


class EKYCFaceMatchResult(BaseModel):
    """Schema for face matching results"""
    is_match: bool = Field(..., description="Whether faces match")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    threshold_used: float = Field(..., description="Threshold used for matching")
    confidence_level: str = Field(..., description="Confidence level: LOW, MEDIUM, HIGH")
    error: Optional[str] = Field(None, description="Error message if matching failed")


class EKYCLivenessResult(BaseModel):
    """Schema for liveness detection results"""
    is_live: bool = Field(..., description="Whether image appears to be live")
    score: float = Field(..., description="Liveness confidence score (0-1)")
    reason: Optional[str] = Field(None, description="Reason for liveness decision")
    checks: Dict[str, Any] = Field(..., description="Individual liveness checks performed")


class FaceAnalysis(BaseModel):
    """Schema for face analysis results"""
    faces_detected: int = Field(..., description="Number of faces detected")
    primary_face: Dict[str, Any] = Field(..., description="Primary face information")
    face_quality: Dict[str, Any] = Field(..., description="Face quality assessment")
    embedding_extracted: bool = Field(..., description="Whether embedding was successfully extracted")


class ConfidenceScores(BaseModel):
    """Schema for confidence scores"""
    id_card_quality: float = Field(..., description="ID card image quality score")
    selfie_quality: float = Field(..., description="Selfie image quality score")
    liveness_score: float = Field(..., description="Liveness detection score")
    face_match_score: float = Field(..., description="Face matching score")


class OverallResult(BaseModel):
    """Schema for overall verification result"""
    status: str = Field(..., description="Verification status: PASSED, FAILED, ERROR")
    confidence: float = Field(..., description="Overall confidence score")
    decision_factors: List[str] = Field(..., description="Factors that influenced the decision")
    reason: Optional[str] = Field(None, description="Reason for failure/error")
    error: Optional[str] = Field(None, description="Error message if verification failed")


class EKYCVerificationResult(BaseModel):
    """Schema for complete EKYC verification results"""
    verification_id: str = Field(..., description="Unique verification identifier")
    timestamp: str = Field(..., description="Verification timestamp")
    user_data: Dict[str, Any] = Field(..., description="User data provided")
    id_card_analysis: FaceAnalysis = Field(..., description="ID card image analysis")
    selfie_analysis: FaceAnalysis = Field(..., description="Selfie image analysis")
    face_match: EKYCFaceMatchResult = Field(..., description="Face matching results")
    liveness_check: EKYCLivenessResult = Field(..., description="Liveness detection results")
    overall_result: OverallResult = Field(..., description="Overall verification result")
    confidence_scores: ConfidenceScores = Field(..., description="Individual confidence scores")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")


class FaceQualityAnalysis(BaseModel):
    """Schema for face quality analysis"""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    brightness: float = Field(..., description="Image brightness score")
    contrast: float = Field(..., description="Image contrast score")
    sharpness: float = Field(..., description="Image sharpness score")
    face_size: float = Field(..., description="Face size adequacy score")
    pose_quality: float = Field(..., description="Face pose quality score")
    eye_visibility: float = Field(..., description="Eye visibility score")
    resolution: Dict[str, int] = Field(..., description="Image resolution information")


class EKYCFeedback(BaseModel):
    """Schema for EKYC verification feedback"""
    verification_id: str = Field(..., description="Verification ID to provide feedback for")
    actual_result: bool = Field(..., description="Actual verification result (ground truth)")
    system_result: bool = Field(..., description="System's verification result")
    confidence: float = Field(..., description="System's confidence score")
    user_comments: Optional[str] = Field(None, description="User comments or additional feedback")
    feedback_type: str = Field("accuracy", description="Type of feedback: accuracy, quality, user_experience")


class EKYCSession(BaseModel):
    """Schema for EKYC session information"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Session last update timestamp")
    status: str = Field(..., description="Session status: active, completed, failed, expired")
    verification_results: Optional[List[EKYCVerificationResult]] = Field(None, description="Verification results in this session")


class EKYCStatistics(BaseModel):
    """Schema for EKYC statistics"""
    total_verifications: int = Field(..., description="Total number of verifications")
    successful_verifications: int = Field(..., description="Number of successful verifications")
    failed_verifications: int = Field(..., description="Number of failed verifications")
    average_confidence: float = Field(..., description="Average confidence score")
    success_rate: float = Field(..., description="Success rate percentage")
    common_failure_reasons: List[str] = Field(..., description="Most common failure reasons")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")


class EKYCModelInfo(BaseModel):
    """Schema for EKYC model information"""
    model_type: str = Field(..., description="Type of face recognition model")
    model_version: str = Field(..., description="Model version")
    status: str = Field(..., description="Model status: active, training, updating")
    last_updated: str = Field(..., description="Last update timestamp")
    performance_metrics: Dict[str, float] = Field(..., description="Current performance metrics")
    configuration: Dict[str, Any] = Field(..., description="Model configuration parameters")


class BatchVerificationRequest(BaseModel):
    """Schema for batch verification request"""
    verifications: List[Dict[str, Any]] = Field(..., description="List of verification requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    priority: str = Field("normal", description="Processing priority: low, normal, high")
    callback_url: Optional[str] = Field(None, description="Callback URL for results")


class BatchVerificationResult(BaseModel):
    """Schema for batch verification results"""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of verification requests")
    completed: int = Field(..., description="Number of completed verifications")
    failed: int = Field(..., description="Number of failed verifications")
    results: List[EKYCVerificationResult] = Field(..., description="Individual verification results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    started_at: str = Field(..., description="Batch processing start time")
    completed_at: Optional[str] = Field(None, description="Batch processing completion time")


class EKYCAlert(BaseModel):
    """Schema for EKYC system alerts"""
    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert: security, performance, error")
    severity: str = Field(..., description="Alert severity: low, medium, high, critical")
    message: str = Field(..., description="Alert message")
    timestamp: str = Field(..., description="Alert timestamp")
    verification_id: Optional[str] = Field(None, description="Related verification ID")
    user_id: Optional[str] = Field(None, description="Related user ID")
    metadata: Dict[str, Any] = Field(..., description="Additional alert metadata")


class EKYCHealthCheck(BaseModel):
    """Schema for EKYC system health check"""
    status: str = Field(..., description="Overall system status: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Individual component statuses")
    metrics: Dict[str, float] = Field(..., description="System performance metrics")
    alerts: List[EKYCAlert] = Field(..., description="Active system alerts")
    uptime: float = Field(..., description="System uptime in hours")
    version: str = Field(..., description="System version")
