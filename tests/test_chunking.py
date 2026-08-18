import pytest

pytest.importorskip("langchain_text_splitters")

from rag_ingest.chunking import split_text


def test_split_text_preserves_content_and_multiple_chunks():
    text = "alpha beta gamma. " * 120
    chunks = split_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, chunk_overlap=100)
