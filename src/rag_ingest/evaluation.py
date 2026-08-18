from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import yaml

from .config import get_settings
from .ingestion import ingest_path
from .retrieval import PgVectorRetriever


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant_sources: set[str]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(EvalCase(row["query"], set(row["relevant_sources"])))
    return cases


def retrieval_metrics(results: list[list[str]], cases: list[EvalCase]) -> dict[str, float]:
    if not cases:
        raise ValueError("Evaluation set is empty")
    recalls = []
    reciprocal_ranks = []
    for ranked_sources, case in zip(results, cases, strict=True):
        if not case.relevant_sources:
            raise ValueError("Every evaluation case needs at least one relevant source")
        ranked_set = set(ranked_sources)
        recalls.append(len(ranked_set & case.relevant_sources) / len(case.relevant_sources))

        first_rank = None
        for rank, source in enumerate(ranked_sources, start=1):
            if source in case.relevant_sources:
                first_rank = rank
                break
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
    recall = sum(recalls) / len(recalls)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return {"recall_at_k": recall, "mrr": mrr, "selection_score": 0.6 * mrr + 0.4 * recall}


def _dedupe_sources(chunks) -> list[str]:
    ordered = []
    seen = set()
    for chunk in chunks:
        if chunk.source not in seen:
            ordered.append(chunk.source)
            seen.add(chunk.source)
    return ordered


def run_experiments(session, config_path: str | Path) -> dict:
    settings = get_settings()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    docs_path = Path(config["documents_path"])
    eval_path = Path(config["eval_path"])
    cases = load_eval_cases(eval_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    best = None
    runs = []
    for experiment in config["experiments"]:
        params = {
            "embedding_model": experiment["embedding_model"],
            "chunk_size": int(experiment["chunk_size"]),
            "chunk_overlap": int(experiment["chunk_overlap"]),
            "k": int(experiment.get("k", settings.top_k)),
            "prompt_version": experiment.get("prompt_version", settings.prompt_version),
        }

        ingest_path(
            session,
            docs_path,
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            embedding_model=params["embedding_model"],
        )
        retriever = PgVectorRetriever(
            session,
            **{
                key: params[key]
                for key in ("embedding_model", "chunk_size", "chunk_overlap", "k")
            },
        )
        ranked = [_dedupe_sources(retriever.search(case.query)) for case in cases]
        metrics = retrieval_metrics(ranked, cases)
        details = [
            {
                "query": case.query,
                "relevant_sources": sorted(case.relevant_sources),
                "retrieved_sources": sources,
            }
            for case, sources in zip(cases, ranked, strict=True)
        ]

        with mlflow.start_run(run_name=experiment.get("name")) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(eval_path), artifact_path="evaluation")
            mlflow.log_dict(details, "evaluation/retrieval_results.json")
            run_summary = {"run_id": run.info.run_id, **params, **metrics}

        runs.append(run_summary)
        if best is None or run_summary["selection_score"] > best["selection_score"]:
            best = run_summary

    return {"best": best, "runs": runs}
