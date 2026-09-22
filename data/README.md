# Data foundation

This directory is the **data lake layout** for the Customer Churn Prediction Platform.

It stores datasets. It does not acquire, validate, or train. Those jobs live in `data_acquisition/`, `data_validation/`, `data_preparation/`, and later `ml/`.

## Why raw data is immutable

Raw files are the audit trail. If a training run goes wrong, we must be able to answer "what bytes did we actually receive from IBM, SQLite, or the mock API?" Cleaning, imputation, and feature engineering therefore write **copies** under `interim/` and `processed/`. They never overwrite `raw/`.

## Why processed data is separate

Processed data is a derived product. It can be rebuilt from raw + code. Mixing the two would hide whether a metric change came from a new extract or from a transformation bug.

## Why data is not committed to Git

Git is for code and configuration. CSV/SQLite snapshots are large, often licensed for use rather than redistribution, and they change every extract. Committing them also risks leaking simulated "customer" files into pull requests as if they were production records.

## Future DVC

DVC (or a lakehouse catalog) will version `data/raw` and `data/processed` by checksum and remote pointer. This phase only leaves `.gitignore` hooks and metadata sidecars (`*.metadata.json`) so DVC can be added later without redesigning folders.

## Future object storage

| Current | Production |
| --- | --- |
| Local folders under `data/` | S3 / GCS / Azure Blob with prefix `s3://…/churn/raw/…` |
| Sidecar JSON metadata | Glue/Data Catalog + object tags |
| Developer laptop | Bucket policies, SSE-KMS, lifecycle rules |

## Reproducibility

Every acquired file should have:

- source URL or connection label (never a password)
- UTC timestamp
- SHA-256 checksum
- row/column counts
- catalog id / license notes

Re-running an extract with `overwrite: false` must not silently replace raw bytes.

## Dataset versioning

Until DVC exists, a version is `(catalog_id, sha256, acquired_at_utc)` for public files and `(snapshot_dir, sha256 list)` for database/API extracts. Synthetic snapshots are versioned by `(random_seed, period, drift_factor, sha256)`.

## Layout

```text
data/
  sources/     catalog and local SQLite simulation file (not a warehouse)
  raw/         immutable extracts
  interim/     cleaned copies and validation reports
  processed/   analysis-ready copies (still not model features)
  synthetic/   labeled synthetic snapshots, including time partitions
```

## Commands

See `data_acquisition/README.md` and the root `README.md`.
