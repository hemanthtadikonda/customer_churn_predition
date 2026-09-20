import pandas as pd
import pytest

from ml.data.generate import generate_churn_dataset
from ml.data.schema import FEATURE_COLUMNS, REQUIRED_COLUMNS, SchemaError, validate_dataframe


def test_generated_data_matches_contract():
    df = generate_churn_dataset(n_samples=200, random_state=1)
    validate_dataframe(df)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df["Churn"].isin(["Yes", "No"]).all()
    assert 0.10 < (df["Churn"] == "Yes").mean() < 0.45


def test_schema_rejects_unknown_contract():
    df = generate_churn_dataset(n_samples=20, random_state=1)
    df.loc[0, "Contract"] = "Weekly"
    with pytest.raises(SchemaError, match="Contract"):
        validate_dataframe(df)


def test_schema_rejects_missing_feature_columns():
    df = pd.DataFrame({"customerID": ["CUST1"], "Churn": ["No"]})
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_dataframe(df)


def test_schema_rejects_inconsistent_phone_service():
    df = generate_churn_dataset(n_samples=50, random_state=1)
    no_phone = df.index[df["PhoneService"].eq("No")]
    assert len(no_phone) > 0
    df.loc[no_phone[0], "MultipleLines"] = "Yes"
    with pytest.raises(SchemaError, match="MultipleLines"):
        validate_dataframe(df)


def test_feature_columns_do_not_include_target():
    assert "Churn" not in FEATURE_COLUMNS
    assert "customerID" not in FEATURE_COLUMNS
