from celery import Celery
from app.config import settings

# Celery app configuration
celery_app = Celery(
    'ekyc_tasks',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=['app.tasks.processing_tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True
)
