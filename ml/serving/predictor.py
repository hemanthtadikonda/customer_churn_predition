"""Serving wrapper around the frozen XGBoost sklearn pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.config import load_config, resolve_path
from ml.data.schema import FEATURE_COLUMNS, SchemaError


def risk_band(probability: float) -> str:
    if probability >= 0.60:
        return "high"
    if probability >= 0.30:
        return "medium"
    return "low"


class ChurnPredictor:
    def __init__(
        self,
        pipeline,
        metadata: dict,
        threshold: float,
        model_source: str = "local_artifact",
        model_uri: str | None = None,
    ):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = threshold
        self.model_source = model_source
        self.model_uri = model_uri

    def predict_one(self, payload: dict) -> dict:
        frame = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        missing = [col for col in FEATURE_COLUMNS if col not in payload]
        if missing:
            raise SchemaError(f"Missing features: {missing}")
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": int(probability >= self.threshold),
            "risk_band": risk_band(probability),
            "threshold": self.threshold,
            "model_name": self.metadata.get("model_name"),
            "algorithm": self.metadata.get("algorithm"),
        }

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        return [self.predict_one(row) for row in rows]


def _load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    import json

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_from_mlflow(model_uri: str, config: dict) -> ChurnPredictor:
    from ml.mlflow_utils import configure_mlflow, resolve_tracking_uri

    configure_mlflow(config)
    resolve_tracking_uri(config)  # ensures MLFLOW_ALLOW_FILE_STORE when needed
    try:
        import mlflow.sklearn
    except ImportError as exc:
        raise FileNotFoundError("MLflow is not installed; cannot load MLFLOW_MODEL_URI.") from exc

    pipeline = mlflow.sklearn.load_model(model_uri)
    metadata_path = resolve_path(config["model"]["metadata_path"])
    metadata = _load_metadata(metadata_path)
    metadata.setdefault("model_name", config["model"]["name"])
    metadata.setdefault("algorithm", "XGBoost")
    metadata["mlflow_model_uri"] = model_uri
    threshold = float(metadata.get("threshold", config["model"]["threshold"]))
    return ChurnPredictor(
        pipeline=pipeline,
        metadata=metadata,
        threshold=threshold,
        model_source="mlflow_registry",
        model_uri=model_uri,
    )


def _load_from_joblib(artifact_path: str | None, config: dict) -> ChurnPredictor:
    model_path = (
        Path(artifact_path) if artifact_path else resolve_path(config["model"]["artifact_path"])
    )
    metadata_path = resolve_path(config["model"]["metadata_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run: python -m ml.training.train"
        )
    pipeline = joblib.load(model_path)
    metadata = _load_metadata(metadata_path)
    threshold = float(metadata.get("threshold", config["model"]["threshold"]))
    return ChurnPredictor(
        pipeline=pipeline,
        metadata=metadata,
        threshold=threshold,
        model_source="local_artifact",
        model_uri=str(model_path.as_posix()),
    )


@lru_cache(maxsize=1)
def load_predictor(artifact_path: str | None = None) -> ChurnPredictor:
    config = load_config()
    from ml.mlflow_utils import resolve_serving_model_uri

    model_uri = resolve_serving_model_uri(config)
    if model_uri and artifact_path is None:
        return _load_from_mlflow(model_uri, config)
    return _load_from_joblib(artifact_path, config)
