from contextlib import asynccontextmanager
from typing import Annotated

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from .celery_app import celery
from .config import get_settings
from .db import get_session, init_db
from .rag import ask
from .retrieval import PgVectorRetriever
from .tasks import ingest_documents

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    init_db()
    yield


app = FastAPI(title="rag-ingest", version="0.1.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    path: str = "/data/sample_docs"
    chunk_size: int = Field(default=settings.chunk_size, ge=100, le=4000)
    chunk_overlap: int = Field(default=settings.chunk_overlap, ge=0, le=1000)
    embedding_model: str = settings.embedding_model


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=settings.top_k, ge=1, le=20)
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
    embedding_model: str = settings.embedding_model
    prompt_version: str = settings.prompt_version
    generate: bool = True


@app.get("/health")
def health(session: Annotated[Session, Depends(get_session)]) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/ingest", status_code=202)
def ingest(request: IngestRequest) -> dict:
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be smaller than chunk_size")
    task = ingest_documents.delay(
        request.path,
        request.chunk_size,
        request.chunk_overlap,
        request.embedding_model,
    )
    return {"task_id": task.id, "state": task.state}


@app.get("/tasks/{task_id}")
def task_status(task_id: str) -> dict:
    result = AsyncResult(task_id, app=celery)
    payload = {"task_id": task_id, "state": result.state}
    if result.successful():
        payload["result"] = result.result
    elif result.failed():
        payload["error"] = str(result.result)
    elif isinstance(result.info, dict):
        payload["meta"] = result.info
    return payload


@app.post("/query")
def query(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
) -> dict:
    retriever = PgVectorRetriever(
        session,
        embedding_model=request.embedding_model,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        k=request.k,
    )
    if request.generate:
        try:
            return ask(retriever, request.question, request.prompt_version)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    chunks = retriever.search(request.question)
    return {
        "question": request.question,
        "answer": None,
        "prompt_version": request.prompt_version,
        "sources": [chunk.as_dict() for chunk in chunks],
    }
