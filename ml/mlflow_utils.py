"""MLflow tracking URI resolution, model logging, and registry helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ml.config import resolve_path

# MLflow 3+ defaults to skops; custom steps and XGBoost need explicit trust.
MLFLOW_SKOPS_TRUSTED_TYPES = [
    "ml.features.engineer.ChurnFeatureEngineer",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]

DEFAULT_LOGGED_MODEL_NAME = "sklearn_pipeline"


def resolve_tracking_uri(config: dict) -> str:
    """Prefer MLFLOW_TRACKING_URI; else configs/config.yaml mlflow.tracking_uri."""
    configured = str(config.get("mlflow", {}).get("tracking_uri", "mlruns")).strip()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip() or configured
    if uri.startswith(("http://", "https://")):
        return uri
    tracking_dir = resolve_path(uri)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return tracking_dir.as_uri()


def resolve_serving_model_uri(config: dict) -> str | None:
    """Model URI for inference: env MLFLOW_MODEL_URI, then config mlflow.model_uri."""
    env_uri = os.environ.get("MLFLOW_MODEL_URI", "").strip()
    if env_uri:
        return env_uri
    cfg_uri = str(config.get("mlflow", {}).get("model_uri", "") or "").strip()
    return cfg_uri or None


def tracking_server_required_for_registry(tracking_uri: str) -> bool:
    return tracking_uri.startswith(("http://", "https://"))


def configure_mlflow(config: dict):
    try:
        import mlflow
    except ImportError:
        return None

    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return mlflow


def log_sklearn_pipeline(mlflow, pipeline, artifact_name: str = DEFAULT_LOGGED_MODEL_NAME) -> None:
    try:
        mlflow.sklearn.log_model(
            pipeline,
            name=artifact_name,
            skops_trusted_types=MLFLOW_SKOPS_TRUSTED_TYPES,
        )
    except TypeError:
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=artifact_name,
            skops_trusted_types=MLFLOW_SKOPS_TRUSTED_TYPES,
        )


@dataclass(frozen=True)
class RegistryResult:
    run_id: str
    registered_model_name: str
    version: str
    model_uri: str
    alias: str | None


def registry_settings(config: dict) -> dict[str, Any]:
    mlflow_cfg = config.get("mlflow", {}) or {}
    registry = mlflow_cfg.get("registry", {}) or {}
    env_register = os.environ.get("MLFLOW_REGISTER_MODEL", "").strip().lower()
    enabled = registry.get("enabled", True)
    if env_register in {"0", "false", "no"}:
        enabled = False
    elif env_register in {"1", "true", "yes"}:
        enabled = True
    return {
        "enabled": bool(enabled),
        "registered_model_name": str(
            registry.get("registered_model_name") or mlflow_cfg.get("experiment_name") or "churn-model"
        ),
        "logged_model_name": str(registry.get("logged_model_name") or DEFAULT_LOGGED_MODEL_NAME),
        "promote_alias_on_pass": _optional_str(registry.get("promote_alias_on_pass")),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def register_run_model(
    config: dict,
    run_id: str,
    tracking_uri: str,
) -> RegistryResult | None:
    """Register the logged sklearn model and optionally assign a registry alias."""
    settings = registry_settings(config)
    if not settings["enabled"]:
        return None
    if not tracking_server_required_for_registry(tracking_uri):
        print(
            "MLflow model registry skipped: use an HTTP(S) tracking server "
            "(MLFLOW_TRACKING_URI), not a local file store."
        )
        return None

    import mlflow
    from mlflow import MlflowClient

    name = settings["registered_model_name"]
    artifact = settings["logged_model_name"]
    source_uri = f"runs:/{run_id}/{artifact}"

    mlflow.set_tracking_uri(tracking_uri)
    registered = mlflow.register_model(model_uri=source_uri, name=name)
    version = str(registered.version)
    model_uri = f"models:/{name}/{version}"

    alias = settings["promote_alias_on_pass"]
    if alias:
        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_registered_model_alias(name=name, alias=alias, version=version)
        model_uri = f"models:/{name}@{alias}"

    return RegistryResult(
        run_id=run_id,
        registered_model_name=name,
        version=version,
        model_uri=model_uri,
        alias=alias,
    )
