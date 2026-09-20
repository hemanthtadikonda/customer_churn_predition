"""Data contract for the telco churn dataset.

This schema is the shared language between data science, the training
pipeline, and the serving API. Changing it is a breaking contract change.
"""

from __future__ import annotations

import pandas as pd

ID_COL = "customerID"
TARGET_COL = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

DERIVED_NUMERIC_FEATURES = [
    "avg_monthly_spend",
    "services_count",
    "tenure_group",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COL] + FEATURE_COLUMNS + [TARGET_COL]

ALLOWED_VALUES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET_COL: {POSITIVE_LABEL, NEGATIVE_LABEL},
}

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class SchemaError(ValueError):
    """Raised when a dataframe does not match the churn data contract."""


def _validate_service_consistency(df: pd.DataFrame) -> None:
    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        phone_mismatch = df["PhoneService"].eq("No") & ~df["MultipleLines"].eq("No phone service")
        if phone_mismatch.any():
            raise SchemaError("MultipleLines must be 'No phone service' when PhoneService is No")

    if "InternetService" in df.columns and all(col in df.columns for col in SERVICE_COLUMNS):
        no_internet = df["InternetService"].eq("No")
        for col in SERVICE_COLUMNS:
            mismatch = no_internet & ~df[col].eq("No internet service")
            if mismatch.any():
                raise SchemaError(f"{col} must be 'No internet service' when InternetService is No")


def encode_target(series: pd.Series) -> pd.Series:
    return series.map({POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0}).astype(int)


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> None:
    required = REQUIRED_COLUMNS if require_target else [ID_COL] + FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    extra = sorted(set(df.columns) - set(REQUIRED_COLUMNS))
    if extra:
        raise SchemaError(f"Unexpected columns: {extra}")

    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        if col == "SeniorCitizen":
            values = set(df[col].dropna().astype(int).unique())
        else:
            values = set(df[col].dropna().unique())
        unknown = values - allowed
        if unknown:
            raise SchemaError(f"Unexpected values in {col}: {sorted(unknown)}")

    _validate_service_consistency(df)
