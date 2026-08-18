from celery import Celery

from .config import get_settings

settings = get_settings()
celery = Celery(
    "rag_ingest",
    broker=settings.redis_url,
    backend=settings.celery_result_backend,
    include=["rag_ingest.tasks"],
)
celery.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
)
