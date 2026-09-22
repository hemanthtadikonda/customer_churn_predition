import pandas as pd

from data_acquisition.synthetic_source import generate_operational_frame
from data_validation.quality_report import render_quality_report
from data_validation.validate import validate_frame


def test_valid_synthetic_data_passes():
    frame = generate_operational_frame(n_records=300, random_seed=4, period="2026-01")
    report = validate_frame(frame, "synthetic_operational")
    assert report["status"] in {"PASS", "WARN"}
    assert all(check["status"] != "FAIL" for check in report["checks"] if check["name"] == "required_columns")
    assert next(check for check in report["checks"] if check["name"] == "unique_customer_ids")["status"] == "PASS"


def test_missing_columns_fail():
    frame = pd.DataFrame({"customer_id": ["CUST1"], "churn": [0]})
    report = validate_frame(frame, "synthetic_operational")
    assert report["status"] == "FAIL"
    required = next(check for check in report["checks"] if check["name"] == "required_columns")
    assert required["status"] == "FAIL"


def test_duplicate_ids_fail():
    frame = generate_operational_frame(n_records=40, random_seed=1)
    frame.loc[1, "customer_id"] = frame.loc[0, "customer_id"]
    report = validate_frame(frame, "synthetic_operational")
    unique = next(check for check in report["checks"] if check["name"] == "unique_customer_ids")
    assert unique["status"] == "FAIL"


def test_illegal_category_and_impossible_age_fail():
    frame = generate_operational_frame(n_records=30, random_seed=2)
    frame.loc[0, "contract_type"] = "Weekly"
    frame.loc[1, "age"] = -5
    report = validate_frame(frame, "synthetic_operational")
    assert report["status"] == "FAIL"
    allowed = next(check for check in report["checks"] if check["name"] == "allowed_categorical_values")
    ranges = next(check for check in report["checks"] if check["name"] == "numeric_ranges")
    assert allowed["status"] == "FAIL"
    assert ranges["status"] == "FAIL"


def test_degenerate_target_fails():
    frame = generate_operational_frame(n_records=20, random_seed=8)
    frame["churn"] = 0
    report = validate_frame(frame, "synthetic_operational")
    target = next(check for check in report["checks"] if check["name"] == "target_distribution")
    assert target["status"] == "FAIL"


def test_quality_report_renders_status():
    frame = generate_operational_frame(n_records=50, random_seed=6)
    report = validate_frame(frame, "synthetic_operational")
    text = render_quality_report(report)
    assert "Status:" in text
    assert "unique_customer_ids" in text
