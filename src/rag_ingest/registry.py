from __future__ import annotations

import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

from .config import get_settings


class RetrievalConfigModel(PythonModel):
    """Registry-compatible wrapper for the selected retrieval configuration.

    This project does not claim to train a new embedding model. The pyfunc stores the
    winning retrieval settings so the MLflow registry provides versioning and lineage.
    """

    def __init__(self, config: dict):
        self.config = config

    def predict(self, context, model_input, params=None):  # noqa: ARG002
        count = len(model_input) if hasattr(model_input, "__len__") else 1
        payload = json.dumps(self.config, sort_keys=True)
        return pd.Series([payload] * count)


def register_best_config(best: dict) -> dict:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    with mlflow.start_run(run_id=best["run_id"]):
        info = mlflow.pyfunc.log_model(
            name="retrieval_config",
            python_model=RetrievalConfigModel(best),
            input_example=pd.DataFrame({"query": ["example"]}),
            registered_model_name=settings.mlflow_registered_model_name,
        )

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.mlflow_registered_model_name}'")
    newest = max(versions, key=lambda item: int(item.version))
    client.set_registered_model_alias(
        settings.mlflow_registered_model_name,
        "champion",
        newest.version,
    )
    return {
        "registered_model": settings.mlflow_registered_model_name,
        "version": newest.version,
        "alias": "champion",
        "model_uri": info.model_uri,
    }
