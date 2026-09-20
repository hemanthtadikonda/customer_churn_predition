"""Load, validate, and split the collected churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import load_config, resolve_path
from ml.data.schema import (
    FEATURE_COLUMNS,
    ID_COL,
    TARGET_COL,
    encode_target,
    validate_dataframe,
)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    config = load_config()
    csv_path = Path(path) if path else resolve_path(config["data"]["raw_path"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. Run: python -m ml.data.generate"
        )
    df = pd.read_csv(csv_path)
    validate_dataframe(df, require_target=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = df[FEATURE_COLUMNS].copy()
    target = encode_target(df[TARGET_COL])
    customer_ids = df[ID_COL].copy()
    return features, target, customer_ids


def train_val_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    config = load_config()["data"]
    test_size = config["test_size"] if test_size is None else test_size
    val_size = config["val_size"] if val_size is None else val_size
    random_state = config["random_state"] if random_state is None else random_state

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_train_full,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
