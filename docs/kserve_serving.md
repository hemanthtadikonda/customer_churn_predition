# KServe — serve the MLflow Staging model

This guide deploys the same sklearn pipeline used by FastAPI (`ml/serving/predictor.py`) on **KServe**, loading **`models:/customer-churn-xgboost@Staging`** from your MLflow registry.

## Architecture

```text
  Kubernetes (KServe InferenceService)
       |
       |  MLFLOW_TRACKING_URI + MLFLOW_MODEL_URI
       v
  MLflow tracking server (registry + artifact metadata)
       |
       v
  Artifact store (local mlartifacts on VM, or S3)
```

The serving container does **not** bake in a fixed model file. At startup it calls `mlflow.sklearn.load_model` for the alias you set (`@Staging` or `@Production`).

## Prerequisites

1. A Kubernetes cluster with **KServe** installed ([KServe quickstart](https://kserve.github.io/website/latest/admin/serverless/serverless/)).
2. MLflow tracking server on **HTTP(S)** reachable from pods (replace `127.0.0.1` with a cluster DNS name, Node IP, or in-cluster Service).
3. Pods can read MLflow **artifacts** (same network path as training: local volume mount on the MLflow host, or S3 with credentials in the cluster).
4. A registry version with alias **Staging** (your training flow sets this when the quality gate passes).

## Build the serving image

From the repo root:

```bash
docker build -f mlops/Dockerfile.kserve -t customer-churn-kserve:latest .
```

Load into kind/minikube if needed:

```bash
kind load docker-image customer-churn-kserve:latest
# or: minikube image load customer-churn-kserve:latest
```

## Configure environment

Edit `mlops/kserve/inference-service.yaml`:

| Variable | Purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Tracking + registry API URL **as seen from the pod** |
| `MLFLOW_MODEL_URI` | e.g. `models:/customer-churn-xgboost@Staging` |

If artifacts live on **S3**, add AWS environment variables or a Secret and ensure the MLflow server also uses that bucket as `--default-artifact-root`.

## Deploy

```bash
kubectl apply -f mlops/kserve/inference-service.yaml
kubectl get inferenceservice customer-churn-xgboost
kubectl get pods -l serving.kserve.io/inferenceservice=customer-churn-xgboost
```

Wait until `READY` is `True`. Note the inference URL from `kubectl get inferenceservice` (cluster-dependent).

## Predict (V1 JSON — same features as `/predict`)

```bash
HOST="<inference-url>"
curl -s "${HOST}/v1/models/customer-churn-xgboost:predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
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
    }]
  }' | python -m json.tool
```

Response includes `churn_probability`, `churn_prediction`, `risk_band`, and `threshold` per instance (same as FastAPI).

## Promote Staging → Production

1. In the MLflow UI (or API), set alias **Production** on the reviewed version.
2. Update `MLFLOW_MODEL_URI` to `models:/customer-churn-xgboost@Production` and roll the Deployment:

```bash
kubectl set env deployment/<predictor-deployment> \
  MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Production
```

Or change the YAML and re-apply.

## Local smoke test (no Kubernetes)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Staging
pip install -r mlops/requirements-kserve.txt
python -m ml.serving.kserve_model
```

Then call `http://127.0.0.1:8080` with the curl above (adjust host/port).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Pod `CrashLoopBackOff` on start | MLflow unreachable or bad URI | Fix `MLFLOW_TRACKING_URI`; curl from a debug pod |
| `Unable to download artifacts` | S3 creds or wrong artifact root | Align server artifact root with credentials in serving pod |
| `READY False` | Model load slow or probe too early | Increase `readinessProbe.initialDelaySeconds` |
| Stale model after promotion | Env still `@Staging` | Update `MLFLOW_MODEL_URI` and restart pods |
| Works on laptop, fails in K8s | `127.0.0.1` in URI | Use Service DNS or VM IP reachable from cluster |

## Related docs

- [mlflow_guide.md](mlflow_guide.md) — registry aliases and FastAPI serving
- `ml/serving/kserve_model.py` — KServe entrypoint
- `mlops/docker-compose.yml` — non-Kubernetes FastAPI serving
