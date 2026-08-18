from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def validate_dimension(vector: list[float]) -> None:
    expected = get_settings().embedding_dimension
    if len(vector) != expected:
        raise ValueError(
            f"Embedding dimension {len(vector)} does not match EMBEDDING_DIMENSION={expected}. "
            "Use models with the configured dimension or recreate the vector column."
        )
