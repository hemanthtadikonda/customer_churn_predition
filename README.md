# Customer Churn Prediction Platform

Predict whether a telco customer is likely to leave, using **XGBoost** as the core model.

This repo is structured the way a real team hands work off:

1. **Data scientist** collects data and defines the contract.
2. **ML engineer** packages feature engineering + the model behind an API.
3. **MLOps** versions data/models, tracks experiments, tests, and deploys.

XGBoost is the default here because it is still the workhorse for tabular churn: strong accuracy on structured billing/CRM data, built-in feature importance for explainability, a simple sklearn-compatible artifact for serving, and high interview coverage.

## Project layout

```text
ml/                     # ML code folder
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

Requires Python 3.11+.

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

- Replace the synthetic generator with a warehouse extract and add DVC data tracking (`dvc init`, `dvc add data/raw`).
- Tune the serving threshold for recall vs precision on a cost matrix.
- Add SHAP plots from the frozen pipeline for stakeholder explainability.
- Register the MLflow model and promote Staging → Production.
- Add a GitHub Actions job that retrains and fails the build if the quality gate drops.
