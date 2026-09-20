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
    def __init__(self, pipeline, metadata: dict, threshold: float):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = threshold

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


@lru_cache(maxsize=1)
def load_predictor(artifact_path: str | None = None) -> ChurnPredictor:
    config = load_config()
    model_path = Path(artifact_path) if artifact_path else resolve_path(config["model"]["artifact_path"])
    metadata_path = resolve_path(config["model"]["metadata_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run: python -m ml.training.train"
        )
    pipeline = joblib.load(model_path)
    metadata = {}
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ChurnPredictor(pipeline, metadata, float(config["model"]["threshold"]))
