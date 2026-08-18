from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    if project["name"] != "rag-ingest":
        fail("unexpected project name")

    docs = sorted((ROOT / "sample_docs").glob("*.md"))
    if not 15 <= len(docs) <= 20:
        fail(f"expected 15-20 sample documents, found {len(docs)}")

    eval_rows = []
    for line in (ROOT / "eval/relevance.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            eval_rows.append(json.loads(line))
    if not eval_rows:
        fail("evaluation set is empty")

    for row in eval_rows:
        if not row.get("query") or not row.get("relevant_sources"):
            fail(f"invalid evaluation row: {row}")
        for source in row["relevant_sources"]:
            relative = source.replace("/data/sample_docs/", "sample_docs/")
            if not (ROOT / relative).exists():
                fail(f"evaluation source does not exist: {source}")

    for version in ("v1", "v2"):
        prompt = (ROOT / f"prompts/{version}_query.txt").read_text(encoding="utf-8")
        if "{context}" not in prompt or "{question}" not in prompt:
            fail(f"prompt {version} is missing a required placeholder")

    required = [
        "docker-compose.yml",
        "Dockerfile",
        "src/rag_ingest/api.py",
        "src/rag_ingest/tasks.py",
        "src/rag_ingest/evaluation.py",
        "src/rag_ingest/registry.py",
    ]
    for path in required:
        if not (ROOT / path).exists():
            fail(f"missing required file: {path}")

    print(
        f"project validation passed: {len(docs)} sample docs, "
        f"{len(eval_rows)} evaluation queries, 2 prompt versions"
    )


if __name__ == "__main__":
    main()
