"""Validate datasets without modifying raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_acquisition.common import load_acquisition_config, project_path, utc_now_iso, write_json
from data_validation.schema import get_schema

STATUSES = ("PASS", "WARN", "FAIL")


def validate_frame(
    frame: pd.DataFrame,
    schema_name: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run quality checks and return a machine-readable PASS/WARN/FAIL report."""
    cfg = (config or load_acquisition_config()).get("validation", {})
    schema = get_schema(schema_name)
    checks: list[dict[str, Any]] = []

    checks.append(_required_columns(frame, schema))
    checks.append(_duplicate_records(frame, cfg))
    checks.append(_id_uniqueness(frame, schema))
    checks.append(_allowed_values(frame, schema))
    checks.append(_numeric_ranges(frame, schema))
    checks.append(_null_profile(frame, schema, cfg))
    checks.append(_target_distribution(frame, schema, cfg))
    checks.append(_basic_stats(frame, schema))

    overall = _rollup(checks)
    return {
        "status": overall,
        "schema": schema_name,
        "checked_at_utc": utc_now_iso(),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "checks": checks,
        "note": "Validation never mutates raw data. Failures must be handled in preparation, not by rewriting source files.",
    }


def validate_file(path: Path, schema_name: str, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a CSV/parquet file on disk."""
    frame = _read_table(path)
    report = validate_frame(frame, schema_name, config=config)
    report["input_path"] = str(path)
    return report


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _required_columns(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    missing = [col for col in schema["required_columns"] if col not in frame.columns]
    status = "FAIL" if missing else "PASS"
    return _check("required_columns", status, "Missing columns" if missing else "All required columns present", {"missing": missing})


def _duplicate_records(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    n_dupes = int(frame.duplicated().sum())
    fraction = n_dupes / max(len(frame), 1)
    status = _threshold_status(fraction, float(cfg.get("warn_duplicate_fraction", 0.01)), float(cfg.get("fail_duplicate_fraction", 0.05)))
    return _check("duplicate_records", status, f"{n_dupes} exact duplicate rows ({fraction:.2%})", {"count": n_dupes, "fraction": fraction})


def _id_uniqueness(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    id_col = schema.get("id_column")
    if not id_col:
        return _check("unique_customer_ids", "WARN", "Schema has no identifier column", {})
    if id_col not in frame.columns:
        return _check("unique_customer_ids", "FAIL", f"Identifier column {id_col} is missing", {})
    n_dupes = int(frame[id_col].duplicated().sum())
    n_null = int(frame[id_col].isna().sum())
    status = "FAIL" if n_dupes or n_null else "PASS"
    return _check(
        "unique_customer_ids",
        status,
        "Customer IDs are unique" if status == "PASS" else "Duplicate or null customer IDs",
        {"duplicate_ids": n_dupes, "null_ids": n_null},
    )


def _allowed_values(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    unexpected: dict[str, list[Any]] = {}
    for column, allowed in schema.get("allowed_values", {}).items():
        if column not in frame.columns:
            continue
        values = set(_normalize_values(frame[column], allowed))
        unknown = sorted(values - set(allowed), key=str)
        if unknown:
            unexpected[column] = unknown
    status = "FAIL" if unexpected else "PASS"
    return _check("allowed_categorical_values", status, "Unexpected categories" if unexpected else "Categories within allow-list", unexpected)


def _numeric_ranges(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    violations: dict[str, dict[str, Any]] = {}
    for column, bounds in schema.get("numeric_ranges", {}).items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        n_low = int((series < bounds["min"]).sum())
        n_high = int((series > bounds["max"]).sum())
        if n_low or n_high:
            violations[column] = {"below_min": n_low, "above_max": n_high, **bounds}
    status = "FAIL" if violations else "PASS"
    return _check("numeric_ranges", status, "Impossible or out-of-range numeric values" if violations else "Numeric ranges OK", violations)


def _null_profile(frame: pd.DataFrame, schema: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    required = schema["required_columns"]
    present = [col for col in required if col in frame.columns]
    null_fraction = {col: float(frame[col].isna().mean()) for col in present}
    # IBM TotalCharges blanks are expected for tenure=0 and should not fail the extract.
    if schema["name"] == "ibm_telco" and "TotalCharges" in frame.columns:
        coerced = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        null_fraction["TotalCharges"] = float(coerced.isna().mean())
    worst = max(null_fraction.values(), default=0.0)
    status = _threshold_status(worst, float(cfg.get("warn_null_fraction", 0.05)), float(cfg.get("fail_null_fraction", 0.30)))
    if schema["name"] == "ibm_telco" and null_fraction.get("TotalCharges", 0) <= 0.02:
        # Known blanks for new customers; downgrade FAIL on this column alone.
        other = [frac for col, frac in null_fraction.items() if col != "TotalCharges"]
        worst_other = max(other, default=0.0)
        status = _threshold_status(worst_other, float(cfg.get("warn_null_fraction", 0.05)), float(cfg.get("fail_null_fraction", 0.30)))
        if status == "PASS" and null_fraction.get("TotalCharges", 0) > 0:
            status = "WARN"
    return _check("null_percentages", status, f"Highest null fraction={worst:.2%}", null_fraction)


def _target_distribution(frame: pd.DataFrame, schema: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    target = schema["target_column"]
    if target not in frame.columns:
        return _check("target_distribution", "FAIL", f"Target column {target} is missing", {})
    positive = schema["target_positive"]
    rate = float((frame[target] == positive).mean())
    min_rate = float(cfg.get("min_churn_rate", 0.05))
    max_rate = float(cfg.get("max_churn_rate", 0.55))
    if rate < min_rate or rate > max_rate:
        status = "WARN"
        message = f"Churn rate {rate:.1%} is outside expected band {min_rate:.0%}-{max_rate:.0%}"
    else:
        status = "PASS"
        message = f"Churn rate {rate:.1%} within expected band"
    if rate in (0.0, 1.0):
        status = "FAIL"
        message = "Target is degenerate (all one class)"
    return _check("target_distribution", status, message, {"churn_rate": rate, "positive_label": positive})


def _basic_stats(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    numeric_cols = [col for col in schema.get("numeric_ranges", {}) if col in frame.columns]
    summary = {}
    for col in numeric_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        summary[col] = {
            "mean": float(series.mean()) if len(series) else None,
            "std": float(series.std()) if len(series) else None,
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
        }
    return _check("basic_statistics", "PASS", "Numeric summary computed", summary)


def _normalize_values(series: pd.Series, allowed: list[Any]) -> list[Any]:
    if all(isinstance(item, int) and not isinstance(item, bool) for item in allowed):
        return [int(v) for v in pd.to_numeric(series.dropna(), errors="coerce").dropna().tolist()]
    values = []
    for value in series.dropna().tolist():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
        values.append(value)
    return values


def _threshold_status(value: float, warn_at: float, fail_at: float) -> str:
    if value >= fail_at:
        return "FAIL"
    if value >= warn_at:
        return "WARN"
    return "PASS"


def _rollup(checks: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _check(name: str, status: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid status {status}")
    return {"name": name, "status": status, "message": message, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset without modifying it.")
    parser.add_argument("--input", required=True, help="CSV or parquet path")
    parser.add_argument("--schema", default="synthetic_operational", choices=["ibm_telco", "synthetic_operational", "uci_iranian_churn"])
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when overall status is FAIL")
    args = parser.parse_args()
    report = validate_file(Path(args.input), args.schema)
    cfg = load_acquisition_config()
    output = Path(args.output) if args.output else project_path(cfg["validation"]["reports_dir"]) / f"{Path(args.input).stem}_validation.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output)}, indent=2))
    if args.strict and report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
