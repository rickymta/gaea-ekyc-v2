from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_active_user
from app.schemas import TaskStatus, ErrorResponse, User
from app.config import settings
import logging
import redis
import time

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/health",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service unavailable"}
    },
    summary="Health Check",
    description="Kiểm tra tình trạng sức khỏe của service và các dependency"
)
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "service": settings.app_name,
        "version": settings.version,
        "checks": {}
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis connection
    try:
        redis_client = redis.Redis.from_url(settings.redis_url)
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check MinIO connection
    try:
        from app.services.storage_service import storage_manager
        storage_manager.client.list_buckets()
        health_status["checks"]["storage"] = "healthy"
    except Exception as e:
        health_status["checks"]["storage"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status


@router.get(
    "/metrics",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Service Metrics",
    description="Lấy metrics và thống kê của service"
)
async def get_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get service metrics"""
    try:
        from app.models import EKYCSession, EKYCAsset
        
        # Get session statistics
        total_sessions = db.query(EKYCSession).count()
        pending_sessions = db.query(EKYCSession).filter(
            EKYCSession.status == "pending"
        ).count()
        completed_sessions = db.query(EKYCSession).filter(
            EKYCSession.status == "completed"
        ).count()
        failed_sessions = db.query(EKYCSession).filter(
            EKYCSession.status == "failed"
        ).count()
        
        # Get asset statistics
        total_assets = db.query(EKYCAsset).count()
        processed_assets = db.query(EKYCAsset).filter(
            EKYCAsset.processed == True
        ).count()
        unprocessed_assets = db.query(EKYCAsset).filter(
            EKYCAsset.processed == False
        ).count()
        
        return {
            "timestamp": int(time.time()),
            "sessions": {
                "total": total_sessions,
                "pending": pending_sessions,
                "completed": completed_sessions,
                "failed": failed_sessions
            },
            "assets": {
                "total": total_assets,
                "processed": processed_assets,
                "unprocessed": unprocessed_assets
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.get(
    "/tasks/{task_id}/status",
    response_model=TaskStatus,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="Get Task Status",
    description="Kiểm tra trạng thái của một background task"
)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get status of a background task"""
    try:
        from app.tasks.celery_app import celery_app
        
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = {
                "task_id": task_id,
                "status": "pending",
                "result": None,
                "error": None
            }
        elif result.state == 'SUCCESS':
            response = {
                "task_id": task_id,
                "status": "success",
                "result": result.result,
                "error": None
            }
        elif result.state == 'FAILURE':
            response = {
                "task_id": task_id,
                "status": "failed",
                "result": None,
                "error": str(result.info)
            }
        else:
            response = {
                "task_id": task_id,
                "status": result.state.lower(),
                "result": result.info if hasattr(result, 'info') else None,
                "error": None
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task status"
        )


@router.get(
    "/system/info",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"}
    },
    summary="System Information",
    description="Lấy thông tin hệ thống và cấu hình"
)
async def get_system_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get system information"""
    import sys
    import platform
    
    return {
        "service": {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment
        },
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor()
        },
        "configuration": {
            "debug_mode": settings.debug,
            "log_level": settings.log_level,
            "face_match_threshold": settings.face_match_threshold,
            "liveness_threshold": settings.liveness_threshold
        }
    }
