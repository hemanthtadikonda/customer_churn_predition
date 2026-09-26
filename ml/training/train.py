"""Train an XGBoost churn model and persist a serving-ready sklearn pipeline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.config import load_config, resolve_path
from ml.data.generate import generate_churn_dataset, save_dataset
from ml.data.ingest import load_raw, split_features_target, train_val_test_split
from ml.features.engineer import build_feature_pipeline
from ml.training.evaluate import compute_metrics, quality_gate, save_json


def _maybe_start_mlflow(config: dict):
    # Recent MLflow releases refuse the local directory backend unless this
    # is set. The project tracks runs under mlruns/, so keep that store.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError:
        return None

    tracking_dir = resolve_path(config["mlflow"]["tracking_uri"])
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return mlflow


def train_pipeline(
    x_train,
    y_train,
    xgb_params: dict,
) -> Pipeline:
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = dict(xgb_params)
    params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**params)
    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def _feature_importance(pipeline: Pipeline) -> dict[str, float]:
    preprocess = pipeline.named_steps["features"].named_steps["preprocess"]
    names = preprocess.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    ranked = sorted(zip(names.tolist(), importances.tolist()), key=lambda item: item[1], reverse=True)
    return {name: round(score, 6) for name, score in ranked}


def _predict_proba(pipeline: Pipeline, features) -> np.ndarray:
    return pipeline.predict_proba(features)[:, 1]


def run_training(config_path: str | None = None) -> dict:
    config = load_config(Path(config_path) if config_path else None)
    raw_path = resolve_path(config["data"]["raw_path"])
    if not raw_path.exists():
        df = generate_churn_dataset(
            n_samples=int(config["data"]["n_samples"]),
            random_state=int(config["data"]["random_state"]),
        )
        save_dataset(df, raw_path)

    raw = load_raw(raw_path)
    features, target, _ids = split_features_target(raw)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(features, target)

    pipeline = train_pipeline(x_train, y_train, config["model"]["params"])
    threshold = float(config["model"]["threshold"])

    val_metrics = compute_metrics(y_val, _predict_proba(pipeline, x_val), threshold)
    test_metrics = compute_metrics(y_test, _predict_proba(pipeline, x_test), threshold)
    quality_gate(test_metrics)

    artifact_path = resolve_path(config["model"]["artifact_path"])
    metadata_path = resolve_path(config["model"]["metadata_path"])
    metrics_path = resolve_path(config["model"]["metrics_path"])
    importance_path = resolve_path(config["model"]["importance_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifact_path)
    importance = _feature_importance(pipeline)
    metadata = {
        "model_name": config["model"]["name"],
        "algorithm": "XGBoost",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "churn_rate_train": float(y_train.mean()),
        "threshold": threshold,
        "params": config["model"]["params"],
        "scale_pos_weight": float(pipeline.named_steps["model"].get_params()["scale_pos_weight"]),
        "artifact_path": str(artifact_path.as_posix()),
    }
    save_json(metadata, metadata_path)
    save_json({"validation": val_metrics, "test": test_metrics}, metrics_path)
    save_json(importance, importance_path)

    mlflow = _maybe_start_mlflow(config)
    if mlflow is not None:
        with mlflow.start_run(run_name="xgboost_churn"):
            mlflow.log_params(config["model"]["params"])
            mlflow.log_metrics(
                {
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_f1": val_metrics["f1"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_f1": test_metrics["f1"],
                    "test_recall": test_metrics["recall"],
                }
            )
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(importance_path))
            try:
                mlflow.sklearn.log_model(pipeline, artifact_path="sklearn_pipeline")
            except Exception as exc:
                print(f"MLflow model logging skipped: {exc}")

    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    print(f"Saved pipeline to {artifact_path}")
    return {"pipeline": pipeline, "validation": val_metrics, "test": test_metrics, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost churn model.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
