from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .embeddings import get_embedder, validate_dimension
from .models import Chunk, Document


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source: str
    title: str
    chunk_index: int
    content: str
    score: float

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "score": self.score,
        }


class PgVectorRetriever:
    def __init__(
        self,
        session: Session,
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        k: int,
    ) -> None:
        self.session = session
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.embedder = get_embedder(embedding_model)

    def search(self, question: str) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(question)
        validate_dimension(query_vector)
        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = (
            select(Chunk, Document, distance)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.embedding_model == self.embedding_model,
                Chunk.chunk_size == self.chunk_size,
                Chunk.chunk_overlap == self.chunk_overlap,
            )
            .order_by(distance)
            .limit(self.k)
        )
        rows = self.session.execute(stmt).all()
        return [
            RetrievedChunk(
                chunk_id=str(chunk.id),
                document_id=str(document.id),
                source=document.source,
                title=document.title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=max(0.0, 1.0 - float(dist)),
            )
            for chunk, document, dist in rows
        ]
