import pytest

pytest.importorskip("mlflow")

from rag_ingest.evaluation import EvalCase, retrieval_metrics


def test_retrieval_metrics_compute_recall_and_mrr():
    cases = [
        EvalCase("q1", {"a"}),
        EvalCase("q2", {"z"}),
    ]
    results = [["x", "a", "b"], ["z", "y"]]
    metrics = retrieval_metrics(results, cases)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["selection_score"] == pytest.approx(0.85)


def test_recall_counts_all_relevant_documents():
    cases = [EvalCase("q", {"a", "b"})]
    metrics = retrieval_metrics([["a", "x"]], cases)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(1.0)
