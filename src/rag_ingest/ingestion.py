from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .chunking import split_text
from .embeddings import get_embedder, validate_dimension
from .models import Chunk, Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class IngestSummary:
    documents_seen: int
    documents_written: int
    chunks_written: int

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_seen": self.documents_seen,
            "documents_written": self.documents_written,
            "chunks_written": self.chunks_written,
        }


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
    )


def ingest_path(
    session: Session,
    path: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> IngestSummary:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    embedder = get_embedder(embedding_model)
    files = _iter_files(source_path)
    written_docs = 0
    written_chunks = 0

    for file in files:
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        source = str(file)
        checksum = _checksum(text)
        title = text.splitlines()[0].lstrip("# ").strip() or file.stem
        doc = session.scalar(select(Document).where(Document.source == source))

        if doc is None:
            doc = Document(
                source=source,
                title=title,
                checksum=checksum,
                meta={"suffix": file.suffix},
            )
            session.add(doc)
            session.flush()
            written_docs += 1
        elif doc.checksum != checksum:
            doc.title = title
            doc.checksum = checksum
            doc.meta = {"suffix": file.suffix}
            session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            written_docs += 1

        session.execute(
            delete(Chunk).where(
                Chunk.document_id == doc.id,
                Chunk.embedding_model == embedding_model,
                Chunk.chunk_size == chunk_size,
                Chunk.chunk_overlap == chunk_overlap,
            )
        )

        pieces = split_text(text, chunk_size, chunk_overlap)
        vectors = embedder.embed_documents(pieces)
        if vectors:
            validate_dimension(vectors[0])

        for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
            session.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=index,
                    content=piece,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding=vector,
                )
            )
            written_chunks += 1

    session.commit()
    return IngestSummary(len(files), written_docs, written_chunks)
