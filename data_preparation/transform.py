"""Light transformations between cleaned and processed layers.

No model-specific feature engineering lives here. Encoding, scaling, and
derived model features belong in a later ML pipeline.
"""

from __future__ import annotations

import pandas as pd


def transform_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce obvious numeric columns and attach a processed-layer marker."""
    transformed = frame.copy()
    for column in transformed.columns:
        if column.lower() in {"totalcharges", "total_charges", "monthlycharges", "monthly_charges"}:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
    transformed.attrs["layer"] = "processed"
    return transformed
