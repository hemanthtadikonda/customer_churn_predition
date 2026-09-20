"""Feature engineering that is serialized with the model.

The same transformer runs at train time and at API time, which avoids
training/serving skew — a core MLOps requirement.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data.schema import (
    CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    SERVICE_COLUMNS,
)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans billing fields and adds a small set of domain features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)
        tenure = frame["tenure"].clip(lower=0)
        frame["avg_monthly_spend"] = frame["TotalCharges"] / tenure.replace(0, 1)
        frame["services_count"] = (frame[SERVICE_COLUMNS] == "Yes").sum(axis=1)
        frame["tenure_group"] = pd.cut(
            tenure.clip(upper=72),
            bins=[-0.1, 12, 24, 48, 72],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)
        return frame


def build_preprocessor() -> ColumnTransformer:
    numeric = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", ChurnFeatureEngineer()),
            ("preprocess", build_preprocessor()),
        ]
    )
