from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rag-ingest-retrieval"
    mlflow_registered_model_name: str = "rag-ingest-best-retrieval"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5

    prompt_version: str = "v2"
    prompt_dir: Path = Path("prompts")
    ingest_root: Path = Path("data")

    llm_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
