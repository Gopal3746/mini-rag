from pathlib import Path

from .celery_app import celery
from .config import get_settings
from .db import SessionLocal, init_db
from .ingestion import ingest_path


def _validate_ingest_path(raw_path: str) -> Path:
    settings = get_settings()
    root = settings.ingest_root.resolve()
    path = Path(raw_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path {path} is outside INGEST_ROOT={root}")
    return path


@celery.task(name="rag_ingest.ingest", bind=True)
def ingest_documents(
    self,
    path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> dict:
    init_db()
    safe_path = _validate_ingest_path(path)
    self.update_state(state="PROGRESS", meta={"stage": "embedding"})
    with SessionLocal() as session:
        summary = ingest_path(
            session,
            safe_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
    return summary.as_dict()
