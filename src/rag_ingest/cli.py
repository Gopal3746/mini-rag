import json
from pathlib import Path
from typing import Annotated

import typer
from celery.result import AsyncResult

from .celery_app import celery
from .config import get_settings
from .db import SessionLocal, init_db
from .evaluation import run_experiments
from .rag import ask
from .registry import register_best_config
from .retrieval import PgVectorRetriever
from .tasks import ingest_documents

app = typer.Typer(help="Compact RAG ingestion, retrieval, and evaluation CLI.")
settings = get_settings()
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments.yaml")


@app.command()
def init() -> None:
    """Create the pgvector extension and application tables."""
    init_db()
    typer.echo("database initialized")


@app.command()
def ingest(
    path: str,
    chunk_size: int = typer.Option(settings.chunk_size),
    chunk_overlap: int = typer.Option(settings.chunk_overlap),
    embedding_model: str = typer.Option(settings.embedding_model),
    wait: bool = typer.Option(False, help="Wait for the Celery task and print its result."),
) -> None:
    """Queue asynchronous document ingestion on the Celery worker."""
    task = ingest_documents.delay(path, chunk_size, chunk_overlap, embedding_model)
    typer.echo(json.dumps({"task_id": task.id, "state": task.state}, indent=2))
    if wait:
        typer.echo(json.dumps(task.get(timeout=900), indent=2))


@app.command("task")
def task_status(task_id: str) -> None:
    """Inspect a Celery ingestion task."""
    result = AsyncResult(task_id, app=celery)
    payload = {"task_id": task_id, "state": result.state}
    if result.ready():
        payload["result"] = result.result if result.successful() else str(result.result)
    elif isinstance(result.info, dict):
        payload["meta"] = result.info
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def query(
    question: str,
    k: int = typer.Option(settings.top_k),
    chunk_size: int = typer.Option(settings.chunk_size),
    chunk_overlap: int = typer.Option(settings.chunk_overlap),
    embedding_model: str = typer.Option(settings.embedding_model),
    prompt_version: str = typer.Option(settings.prompt_version),
    retrieve_only: bool = typer.Option(False),
) -> None:
    """Retrieve matching chunks and optionally generate a grounded answer."""
    init_db()
    with SessionLocal() as session:
        retriever = PgVectorRetriever(
            session,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        if retrieve_only:
            payload = {
                "question": question,
                "sources": [chunk.as_dict() for chunk in retriever.search(question)],
            }
        else:
            payload = ask(retriever, question, prompt_version)
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option(exists=True)] = DEFAULT_EXPERIMENT_CONFIG,
    register: Annotated[
        bool,
        typer.Option(help="Register the winning config in MLflow."),
    ] = True,
) -> None:
    """Run retrieval experiments and log recall@k/MRR to MLflow."""
    init_db()
    with SessionLocal() as session:
        summary = run_experiments(session, config)
    if register and summary["best"]:
        summary["registry"] = register_best_config(summary["best"])
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
