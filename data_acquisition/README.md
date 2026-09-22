# Data acquisition

## What it does

Collects data from four **intentionally separate** source types and writes **immutable raw snapshots**:

1. Approved public CSV (`csv_source.py`)
2. Simulated production database (`database_init.py` + `database_source.py`)
3. Local mock operational API (`api_source.py`)
4. Documented synthetic generator (`synthetic_source.py`)

`orchestrator.py` runs selected sources. `download.py` is the CSV CLI. `config.yaml` holds non-secret settings.

## Why it exists

In a company, MLOps rarely owns every upstream system. This package simulates those boundaries so one developer can still operate the lifecycle.

## Who owns it in an enterprise

- Public extracts: data engineering / data platform
- Database snapshots: analytics engineering + DBA
- External APIs: vendor integration / app engineering
- Synthetic data: ML platform, clearly labeled
- Orchestration: MLOps (Airflow/Prefect/Dagster later)

## Inputs

- `data/sources/dataset_catalog.yaml`
- `data_acquisition/config.yaml`
- Environment variables for future PostgreSQL / API tokens (see `.env.example`)

## Outputs

- `data/raw/` public CSV + checksum metadata
- `data/raw/database/<snapshot>/` table dumps
- `data/raw/api/<snapshot>/` raw JSON pages
- `data/synthetic/<YYYY-MM>/` labeled synthetic files

## Failure scenarios

- Catalog entry not approved or license missing
- Empty, truncated, or checksum-mismatched download
- SQLite file missing (init not run)
- Database driver unavailable
- HTTP 4xx/5xx, timeouts, or runaway pagination
- PostgreSQL password missing from the environment

## Security considerations

- No API keys or database passwords in Git or logs
- Connection labels omit passwords
- Default API mode is `mock`; public mode requires env-configured base URL

## Reproducibility

CSV downloads skip existing raw files unless `--overwrite`. Synthetic output is seeded. Database seed uses a configured seed. Metadata records SHA-256.

## How to run it

Linux, from the repo root, after `source .venv/bin/activate`:

```bash
python3 -m data_acquisition.synthetic_source --number-of-records 2000 --random-seed 42
python3 -m data_acquisition.database_init
python3 -m data_acquisition.database_source
python3 -m data_acquisition.api_source
python3 -m data_acquisition.orchestrator --sources synthetic,database,api --init-database
```

Public CSV download needs network access and uses only catalog-approved URLs:

```bash
python3 -m data_acquisition.download
```

## Current vs production

| Concern | Current | Production |
| --- | --- | --- |
| Public data | HTTPS GET of an approved CSV | Ingest job from approved zone / vendor feed |
| Database | SQLite file | PostgreSQL / Aurora / RDS |
| API | In-process mock HTTP server | Authenticated vendor API + secret manager |
| Storage | Local `data/raw` | S3 + DVC or a lakehouse table |
| Orchestration | CLI module | Airflow / Dagster / scheduled Kubernetes Job |
