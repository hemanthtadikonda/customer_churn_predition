# Data preparation

## What it does

Provides the **framework** for moving data through layers:

```text
raw  →  validated (report only)  →  cleaned (interim)  →  processed
```

It does **not** perform model-specific feature engineering (no one-hot encoding, scaling, or leakage-prone aggregates for XGBoost).

## Why it exists

Cleaning is a data-engineering concern. Feature engineering is an ML concern. Mixing them makes it unclear whether a production score used the same transformations as training.

## Who owns it in an enterprise

Data engineering owns cleaning/conformance. ML engineering owns feature pipelines (Feast/Tecton or a versioned sklearn pipeline).

## Layer meanings

| Layer | Meaning | Mutable? |
| --- | --- | --- |
| `raw` | Bytes as received | No |
| `validated` | Same bytes plus a quality report | Report only |
| `cleaned` | Trim, duplicate drop, empty-token standardization | New file in `interim/` |
| `processed` | Type coercion and analysis-ready copy | New file in `processed/` |

## Inputs

A path to a raw or synthetic table.

## Outputs

- `data/interim/<name>.cleaned.csv`
- `data/processed/<name>.processed.csv`
- `data/processed/<name>.processed.lineage.json`

## Failure scenarios

- Input path missing
- Unreadable CSV/parquet
- Accidental attempt to write back to `data/raw` (this code does not do that)

## Security considerations

Processed copies still may contain customer-like identifiers. They stay gitignored.

## Reproducibility

Lineage JSON records input checksum and row counts. Cleaning is deterministic.

## How to run it

```bash
python -m data_preparation.prepare --input data/synthetic/2026-01/customers.csv
```

## Current vs production

| Current | Production |
| --- | --- |
| Pandas copies on disk | Spark/dbt models + partitioned lake tables |
| No feature store | Feast / Tecton / warehouse feature tables |
| No training split here | Splits happen in the ML pipeline after this layer |
