# Data validation

## What it does

Checks a dataset against a named schema and emits a machine-readable **PASS / WARN / FAIL** report. It never "fixes" raw files.

## Why it exists

Bad data should fail a pipeline before training. Silent coercion hides contract breaks.

## Who owns it in an enterprise

Analytics engineering / data quality, with MLOps wiring the check into CI and training gates. Tools in production include Great Expectations, Deequ, or dbt tests.

## Inputs

- A CSV/parquet file (raw or synthetic)
- A schema name: `ibm_telco`, `synthetic_operational`, `uci_iranian_churn`
- Thresholds in `data_acquisition/config.yaml` under `validation`

## Outputs

- JSON report under `data/interim/validation_reports/`
- Terminal quality report via `quality_report.py`

## Failure scenarios

- Missing required columns
- Duplicate customer IDs
- Illegal categories
- Numeric values outside documented ranges
- Degenerate target (0% or 100% churn)
- High null rates

## Security considerations

Reports may include column names and aggregate rates, not credentials. Do not dump raw row PII into logs in a real deployment.

## Reproducibility

The same file and schema must produce the same status. Thresholds are config, not hardcoded magic in call sites.

## How to run it

```bash
python -m data_validation.validate --input data/synthetic/2026-01/customers.csv --schema synthetic_operational --strict
python -m data_validation.quality_report --input data/synthetic/2026-01/customers.csv --schema synthetic_operational
```

## Current vs production

| Current | Production |
| --- | --- |
| Hand-written checks | Great Expectations / Deequ / soda |
| JSON sidecar | Quality warehouse table + alerting |
| CLI exit codes | Airflow sensor / CI quality gate |
