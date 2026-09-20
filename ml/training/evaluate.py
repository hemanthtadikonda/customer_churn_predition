"""Classification metrics used as the training quality gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import resolve_path


def compute_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": report,
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def quality_gate(metrics: dict[str, Any], min_roc_auc: float = 0.75) -> None:
    if metrics["roc_auc"] < min_roc_auc:
        raise ValueError(
            f"Quality gate failed: roc_auc={metrics['roc_auc']:.3f} < {min_roc_auc}"
        )


def write_metrics(metrics: dict[str, Any], path: str | Path | None = None) -> Path:
    from ml.config import load_config

    config = load_config()
    output = Path(path) if path else resolve_path(config["model"]["metrics_path"])
    save_json(metrics, output)
    return output
