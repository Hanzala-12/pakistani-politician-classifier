"""
Lightweight MLflow utilities for centralising tracking URI and experiment configuration.

Usage:
    from src.mlflow_utils import configure_mlflow
    configure_mlflow("Pakistani-Politician-Classifier")

This module reads `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` from the
environment and configures mlflow accordingly. It also provides a helper to
register a model if `MLFLOW_REGISTER_MODEL` is set.
"""
import os
import mlflow
import mlflow.pytorch


def configure_mlflow(experiment_name: str = "Pakistani-Politician-Classifier"):
    """Configure MLflow tracking uri and set the experiment.

    Looks for `MLFLOW_TRACKING_URI` env var and uses it if present; otherwise
    defaults to local `mlruns/` (mlflow default). Also honours
    `MLFLOW_EXPERIMENT_NAME` if provided.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp = os.environ.get("MLFLOW_EXPERIMENT_NAME") or experiment_name
    try:
        mlflow.set_experiment(exp)
    except Exception:
        # Best-effort: if setting experiment fails (e.g., remote not available), continue
        pass


def register_model_if_requested(artifact_uri: str, model_name: str):
    """Register a model into MLflow Model Registry if configured.

    Requires `MLFLOW_REGISTER_MODEL=true` and a reachable tracking server.
    """
    do_register = os.environ.get("MLFLOW_REGISTER_MODEL", "false").lower() in ("1", "true", "yes")
    if not do_register:
        return None

    try:
        mlflow.register_model(artifact_uri, model_name)
        return True
    except Exception:
        return None
