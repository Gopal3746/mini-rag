from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")

from rag_ingest.config import get_settings
from rag_ingest.prompts import load_prompt


def test_prompt_contains_required_placeholders(tmp_path: Path, monkeypatch):
    (tmp_path / "v9_query.txt").write_text("Context: {context}\nQ: {question}", encoding="utf-8")
    settings = get_settings()
    monkeypatch.setattr(settings, "prompt_dir", tmp_path)
    assert "{context}" in load_prompt("v9")
