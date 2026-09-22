# Customer Churn Prediction Platform

End-to-end portfolio for telco churn, starting with a **production-style data foundation**. Model training still exists from an earlier step; this phase does not expand it.

The intended lifecycle is:

Data acquisition → validation → versioning → preparation → feature engineering → training → experiment tracking → registry → CI/CD → containers → Kubernetes → serving → monitoring → drift → retraining.

**This phase implements the data foundation only.**

## Data foundation (current phase)

```text
data/                   # lake layout: raw / interim / processed / synthetic / sources
data_acquisition/       # public CSV, SQLite simulation, mock API, synthetic snapshots
data_validation/        # PASS / WARN / FAIL reports; never mutates raw files
data_preparation/       # cleaned + processed copies; no model features yet
docs/                   # architecture, sources, dictionary, lineage, decisions
```

Public samples, synthetic data, and simulated operational systems are kept distinct. See `docs/architecture.md` and `docs/decisions.md`.

### Initialize and acquire

Requires Python 3.12+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m data_acquisition.synthetic_source
python -m data_acquisition.database_init
python -m data_acquisition.database_source
python -m data_acquisition.api_source
python -m data_validation.validate --input data/synthetic/2026-01/customers.csv --schema synthetic_operational
python -m data_preparation.prepare --input data/synthetic/2026-01/customers.csv
pytest -q
```

Networked public CSV (catalog-approved IBM Telco sample only):

```bash
python -m data_acquisition.download
```

Do not commit `data/raw`, `data/synthetic`, or a real `.env`. Copy `.env.example` if you later switch the database driver to PostgreSQL.

## Existing ML path (unchanged this phase)

Predict whether a telco customer is likely to leave, using **XGBoost** as the core model.

This repo is structured the way a real team hands work off:

1. **Data scientist** collects data and defines the contract.
2. **ML engineer** packages feature engineering + the model behind an API.
3. **MLOps** versions data/models, tracks experiments, tests, and deploys.

XGBoost is the default here because it is still the workhorse for tabular churn: strong accuracy on structured billing/CRM data, built-in feature importance for explainability, a simple sklearn-compatible artifact for serving, and high interview coverage.

## Project layout

```text
data/                   # immutable raw, interim, processed, synthetic, source catalog
data_acquisition/       # source adapters + orchestrator
data_validation/        # schema and quality reports
data_preparation/       # cleaning framework
docs/                   # data architecture documentation
ml/                     # ML code folder (existing)
  data/                 # collection, schema, train/val/test split
  features/             # feature engineering shipped with the model
  training/             # XGBoost train + quality gate
  serving/              # load artifact and score customers
api/                    # FastAPI service (ML engineer)
configs/                # training and serving config
data/raw/               # collected dataset (generated locally)
artifacts/              # model, metrics, feature importance
mlops/                  # Docker image and compose file
dvc.yaml                # reproducible collect -> train pipeline
.github/workflows/      # CI tests
```

The same `ChurnFeatureEngineer` runs during training and at `/predict`. That is intentional: it prevents training/serving skew.

## Quickstart

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m ml.data.generate
python -m ml.training.train
pytest -q
uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs and POST a customer to `/predict`.

Example body:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.4,
  "TotalCharges": 246.1
}
```

Response includes `churn_probability`, a 0/1 `churn_prediction`, and a `risk_band` (`low` / `medium` / `high`).

## What each stage does

| Role | Code | Responsibility |
| --- | --- | --- |
| Data scientist | `ml/data`, `ml/features` | Schema, collection, derived features (`avg_monthly_spend`, `services_count`, `tenure_group`) |
| ML engineer | `ml/training`, `ml/serving`, `api` | Train XGBoost, freeze a joblib pipeline, serve predictions |
| MLOps | `dvc.yaml`, `mlruns/`, `mlops/`, CI | Reproducible stages, MLflow tracking, Docker, quality gate (`roc_auc >= 0.75`) |

Training logs params and metrics to MLflow under `mlruns/` and writes:

- `artifacts/models/churn_pipeline.joblib`
- `artifacts/models/metadata.json`
- `artifacts/metrics/metrics.json`
- `artifacts/metrics/feature_importance.json`

## Docker

Train first so `artifacts/models/churn_pipeline.joblib` exists, then:

```bash
docker compose -f mlops/docker-compose.yml up --build
```

## Next MLOps practice steps

The data foundation in `data_acquisition/`, `data_validation/`, and `data_preparation/` is the next wiring target for DVC and training.

- Point `dvc.yaml` collect at catalog-approved raw extracts instead of `ml.data.generate`.
- Add DVC remotes (`dvc add data/raw`) once a bucket exists.
- Tune the serving threshold for recall vs precision on a cost matrix.
- Add SHAP plots from the frozen pipeline for stakeholder explainability.
- Register the MLflow model and promote Staging → Production.
- Add a GitHub Actions job that retrains and fails the build if the quality gate drops.
