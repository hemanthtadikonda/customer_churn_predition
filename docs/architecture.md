# Architecture — data foundation

This phase builds the **data plane** of the Customer Churn Prediction Platform. It does not train or serve models.

## Current local architecture

```text
                    ┌─────────────────────────┐
                    │ dataset_catalog.yaml    │
                    │ (policy + provenance)   │
                    └───────────┬─────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │                          │                          │
┌────▼────┐              ┌──────▼──────┐            ┌──────▼──────┐
│ Public  │              │ Local SQLite│            │ Local mock  │
│ CSV     │              │ simulation  │            │ HTTP API    │
│ (IBM /  │              │ (CRM/bill/  │            │ (not a real │
│  UCI)   │              │  usage)     │            │  vendor)    │
└────┬────┘              └──────┬──────┘            └──────┬──────┘
     │                          │                          │
     └──────────────┬───────────┴────────────┬─────────────┘
                    ▼                        ▼
              data/raw/ (immutable)   data/synthetic/ (labeled fake)
                    │
                    ▼
              data_validation  →  PASS | WARN | FAIL
                    │
                    ▼
              data_preparation (framework)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    data/interim/       data/processed/
          │                   │
          └─────────┬─────────┘
                    ▼
           Future ML pipeline
     (training, registry, serving)
```

## Source boundaries

| Source | What it represents | What it is *not* |
| --- | --- | --- |
| Public CSV | Licensed/tutorial research sample | Production warehouse |
| SQLite | App DB / billing replica | Real PostgreSQL |
| Mock API | External operational feed | A live telecom API |
| Synthetic generator | Controllable operational data + drift | Real customers |

An MLOps engineer consumes these contracts. Data Engineering, DBAs, and vendors would own the upstream originals.

## Module map

| Path | Responsibility |
| --- | --- |
| `data/sources/dataset_catalog.yaml` | Allowed public datasets |
| `data_acquisition/csv_source.py` | Approved HTTPS download + checksum |
| `data_acquisition/database_init.py` | Create/seed local SQLite |
| `data_acquisition/database_source.py` | Extract snapshots (SQLite now, PostgreSQL later) |
| `data_acquisition/api_source.py` | Paginated mock API extract |
| `data_acquisition/synthetic_source.py` | Seeded operational generator |
| `data_acquisition/orchestrator.py` | Run selected sources |
| `data_validation/` | Contract checks, no mutation |
| `data_preparation/` | Clean/processed copies only |

## Future production architecture (not built in this phase)

```text
Vendors / app DBs / events
        │
        ▼
  Ingestion (Airflow/Spark)
        │
        ▼
  Landing bucket (S3) ── DVC / lake catalog
        │
        ▼
  Quality gate (GE/Deequ) ── fail the DAG
        │
        ▼
  Conformed warehouse tables
        │
        ▼
  Feature store + training job
        │
        ▼
  MLflow registry → KServe / SageMaker
        │
        ▼
  Monitoring + drift + retraining trigger
```

DVC, Docker, Kubernetes, MLflow expansion, and model training are **out of scope** for this phase even though earlier portfolio files for some of those tools already exist in the repo.
