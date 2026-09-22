"""Cleaning helpers that operate on copies, never on raw files."""

from __future__ import annotations

import pandas as pd


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim strings, drop exact duplicates, and standardize empty tokens.

    This is not model-specific feature engineering and does not impute
    missing values. Imputation is a training-time decision.
    """
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(_trim_string)
    cleaned = cleaned.replace({"": pd.NA})
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def _trim_string(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    return value
