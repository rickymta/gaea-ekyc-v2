"""
Training-related Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class TrainingDataSource(BaseModel):
    """Schema for training data source configuration"""
    type: str = Field(..., description="Source type: folder, database, or api")
    path: Optional[str] = Field(None, description="Path for folder source")
    connection: Optional[str] = Field(None, description="Connection string for database")
    labels: Optional[Dict[str, str]] = Field(None, description="Label mapping")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional source metadata")


class TrainingConfig(BaseModel):
    """Schema for training configuration"""
    validation_split: float = Field(0.2, description="Validation data split ratio")
    augmentation: Optional[Dict[str, Any]] = Field(None, description="Data augmentation settings")
    optimization: Optional[Dict[str, Any]] = Field(None, description="Optimization parameters")
    batch_size: int = Field(32, description="Training batch size")
    epochs: int = Field(10, description="Number of training epochs")
    learning_rate: float = Field(0.001, description="Learning rate")


class TrainingMetrics(BaseModel):
    """Schema for training metrics"""
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    avg_confidence: float = Field(..., description="Average confidence score")
    total_samples: int = Field(..., description="Total number of samples")


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str = Field(..., description="Type of model")
    embedding_size: int = Field(..., description="Size of face embeddings")
    optimal_threshold: float = Field(..., description="Optimal similarity threshold")
    training_samples: int = Field(..., description="Number of training samples")
    validation_samples: int = Field(..., description="Number of validation samples")
    unique_identities: int = Field(..., description="Number of unique identities")


class TrainingResult(BaseModel):
    """Schema for training results"""
    training_id: str = Field(..., description="Unique training identifier")
    start_time: str = Field(..., description="Training start timestamp")
    end_time: str = Field(..., description="Training end timestamp")
    config: TrainingConfig = Field(..., description="Training configuration used")
    metrics: TrainingMetrics = Field(..., description="Training metrics")
    model_info: ModelInfo = Field(..., description="Model information")
    threshold_optimization: Dict[str, Any] = Field(..., description="Threshold optimization results")


class EvaluationMetrics(BaseModel):
    """Schema for evaluation metrics"""
    accuracy: float = Field(..., description="Evaluation accuracy")
    precision: float = Field(..., description="Evaluation precision")
    recall: float = Field(..., description="Evaluation recall")
    f1_score: float = Field(..., description="Evaluation F1 score")
    avg_confidence: float = Field(..., description="Average confidence score")
    total_samples: int = Field(..., description="Total test samples")


class ErrorAnalysis(BaseModel):
    """Schema for error analysis"""
    false_positives: int = Field(..., description="Number of false positives")
    false_negatives: int = Field(..., description="Number of false negatives")
    error_patterns: List[str] = Field(..., description="Common error patterns")
    confidence_distribution: Dict[str, float] = Field(..., description="Confidence score distribution")


class EvaluationResult(BaseModel):
    """Schema for evaluation results"""
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    timestamp: str = Field(..., description="Evaluation timestamp")
    test_dataset: str = Field(..., description="Test dataset used")
    model_config: Dict[str, Any] = Field(..., description="Model configuration")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    confusion_matrix: Dict[str, Any] = Field(..., description="Confusion matrix")
    error_analysis: ErrorAnalysis = Field(..., description="Error analysis")
    recommendations: List[str] = Field(..., description="Performance recommendations")


class OptimizationConfig(BaseModel):
    """Schema for optimization configuration"""
    target_accuracy: float = Field(0.95, description="Target accuracy threshold")
    speed_requirement: str = Field("medium", description="Speed requirement: low, medium, high")
    memory_limit_mb: int = Field(1024, description="Memory limit in MB")
    precision: str = Field("fp32", description="Model precision: fp16, fp32")
    batch_processing: bool = Field(False, description="Enable batch processing")


class ProductionConfig(BaseModel):
    """Schema for production configuration"""
    face_match_threshold: float = Field(..., description="Face matching threshold")
    liveness_threshold: float = Field(..., description="Liveness detection threshold")
    detection_size: tuple = Field(..., description="Face detection image size")
    max_concurrent_requests: int = Field(..., description="Maximum concurrent requests")
    model_precision: str = Field(..., description="Model precision setting")
    batch_processing: bool = Field(..., description="Batch processing enabled")


class PerformanceMetrics(BaseModel):
    """Schema for performance metrics"""
    inference_time_ms: float = Field(..., description="Average inference time in milliseconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    accuracy: float = Field(..., description="Model accuracy")
    throughput_fps: float = Field(..., description="Throughput in frames per second")


class OptimizationResult(BaseModel):
    """Schema for optimization results"""
    optimization_id: str = Field(..., description="Unique optimization identifier")
    timestamp: str = Field(..., description="Optimization timestamp")
    config: OptimizationConfig = Field(..., description="Optimization configuration")
    optimizations_applied: List[str] = Field(..., description="List of optimizations applied")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    production_config: ProductionConfig = Field(..., description="Production-ready configuration")


class FaceEmbedding(BaseModel):
    """Schema for face embedding"""
    person_id: str = Field(..., description="Person identifier")
    embedding: List[float] = Field(..., description="Face embedding vector")
    confidence: float = Field(..., description="Extraction confidence")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RecognitionMatch(BaseModel):
    """Schema for recognition match"""
    person_id: str = Field(..., description="Matched person identifier")
    similarity: float = Field(..., description="Similarity score")
    is_match: bool = Field(..., description="Whether similarity exceeds threshold")
    confidence_level: str = Field(..., description="Confidence level: LOW, MEDIUM, HIGH")


class BatchRecognitionResult(BaseModel):
    """Schema for batch recognition results"""
    query_image: str = Field(..., description="Query image filename")
    matches: List[RecognitionMatch] = Field(..., description="Recognition matches")
    total_database_size: int = Field(..., description="Size of search database")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class FeedbackData(BaseModel):
    """Schema for user feedback"""
    verification_id: str = Field(..., description="Verification ID")
    actual_result: bool = Field(..., description="Actual verification result")
    system_result: bool = Field(..., description="System prediction result")
    confidence: float = Field(..., description="System confidence score")
    user_comments: Optional[str] = Field(None, description="User comments")
    timestamp: str = Field(..., description="Feedback timestamp")


class ContinuousLearningUpdate(BaseModel):
    """Schema for continuous learning update"""
    new_training_data: Optional[List[Dict[str, Any]]] = Field(None, description="New training samples")
    feedback_data: Optional[List[FeedbackData]] = Field(None, description="User feedback data")
    update_config: Optional[Dict[str, Any]] = Field(None, description="Update configuration")


class ModelPerformanceHistory(BaseModel):
    """Schema for model performance history"""
    timestamp: str = Field(..., description="Performance measurement timestamp")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    threshold_used: float = Field(..., description="Threshold used for evaluation")


class DatasetStatistics(BaseModel):
    """Schema for dataset statistics"""
    total_images: int = Field(..., description="Total number of images")
    total_persons: int = Field(..., description="Total number of unique persons")
    unique_labels: int = Field(..., description="Number of unique labels")
    avg_images_per_person: float = Field(..., description="Average images per person")
    dataset_quality: Dict[str, float] = Field(..., description="Dataset quality metrics")
    balance_score: float = Field(..., description="Dataset balance score")


class TrainingProgress(BaseModel):
    """Schema for training progress"""
    training_id: str = Field(..., description="Training identifier")
    status: str = Field(..., description="Training status")
    progress_percentage: float = Field(..., description="Training progress percentage")
    current_epoch: int = Field(..., description="Current training epoch")
    total_epochs: int = Field(..., description="Total training epochs")
    current_metrics: Optional[TrainingMetrics] = Field(None, description="Current metrics")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")


class ModelComparison(BaseModel):
    """Schema for model comparison"""
    model_a_id: str = Field(..., description="First model identifier")
    model_b_id: str = Field(..., description="Second model identifier")
    comparison_metrics: Dict[str, float] = Field(..., description="Comparison metrics")
    performance_difference: Dict[str, float] = Field(..., description="Performance differences")
    recommendation: str = Field(..., description="Recommendation based on comparison")
    significance_test: Optional[Dict[str, Any]] = Field(None, description="Statistical significance test results")
