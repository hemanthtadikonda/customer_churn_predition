# KServe integration — serve the MLflow Staging model

End-to-end guide for this repo: **kind** cluster on a Linux VM, **KServe 0.20 (Standard mode)**, **MLflow** on the host, and the custom image in `mlops/Dockerfile.kserve`.

The predictor loads the same sklearn pipeline as FastAPI (`ml/serving/predictor.py`) via **`MLFLOW_MODEL_URI`** (e.g. `models:/customer-churn-xgboost@Staging`).

## What lives in the repo

| Path | Role |
| --- | --- |
| `ml/serving/kserve_model.py` | KServe `ModelServer` entrypoint |
| `ml/serving/kserve_handlers.py` | V1 `instances` / V2 tensor request parsing |
| `mlops/Dockerfile.kserve` | Serving image (port **8080**) |
| `mlops/requirements-kserve.txt` | `kserve` Python package for the image |
| `mlops/kserve/inference-service.yaml` | Default `InferenceService` (edit MLflow URI) |
| `mlops/kserve/inference-service-kind.yaml` | kind + MLflow-on-host overlay (`envsubst`) |

## Architecture

```text
  kind cluster (e.g. name: mlops)
       |
       v
  InferenceService customer-churn-xgboost (Standard mode)
       |  env: MLFLOW_TRACKING_URI, MLFLOW_MODEL_URI
       v
  Pod: customer-churn-kserve:latest
       |  mlflow.sklearn.load_model(registry)
       v
  MLflow on VM (~/mlflow-setup) + artifacts (disk or S3)
```

Pods must reach MLflow at a **host-reachable** URL. `http://127.0.0.1:5000` inside the pod is wrong — that is the pod, not the VM.

## Prerequisites

1. **kind** cluster with `kubectl` context set (this project often uses cluster name **`mlops`**, node `mlops-control-plane`).
2. **MLflow** tracking server on HTTP(S), model registered with alias **Staging** ([mlflow_guide.md](mlflow_guide.md)).
3. **Docker** on the VM to build `customer-churn-kserve:latest`.

Confirm context and cluster name:

```bash
kubectl config current-context    # e.g. kind-mlops
kind get clusters                 # e.g. mlops
kubectl get nodes
```

---

## Step 1 — Install KServe (Standard mode)

The old `hack/quick_install.sh` URL on GitHub **404s**; use the **release install script** (tested with **v0.20.0**):

```bash
curl -fL \
  https://github.com/kserve/kserve/releases/download/v0.20.0/kserve-standard-mode-full-install-with-manifests.sh \
  -o /tmp/kserve-install.sh
bash /tmp/kserve-install.sh
```

The installer pulls **cert-manager**, KServe **CRDs**, the controller in namespace `kserve`, and sets **`deploy.defaultDeploymentMode=Standard`** (no Knative/Istio required for lab use).

Verify before creating an `InferenceService`:

```bash
kubectl get crd inferenceservices.serving.kserve.io
kubectl get pods -n kserve
```

Expected: CRD exists; `kserve-controller-manager` is **Running**.

If `kubectl apply` fails with *no matches for kind InferenceService*, KServe is not installed yet — repeat this step.

---

## Step 2 — MLflow on the host (reachable from pods)

On the VM (separate venv under `~/mlflow-setup`), start the server so **pods on the Docker bridge can connect**:

```bash
cd ~/mlflow-setup
source .venv/bin/activate
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --allowed-hosts '*'
```

From the VM shell, `curl http://127.0.0.1:5000` should work. From a cluster pod you will use the **host gateway IP**, not `127.0.0.1`:

```bash
export MLFLOW_HOST_IP="$(ip -4 route show default | awk '{print $3}')"
echo "$MLFLOW_HOST_IP"
curl -s "http://${MLFLOW_HOST_IP}:5000/" | head
```

Train and register if needed (from this repo):

```bash
cd ~/customer_churn_predition
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m ml.training.train
```

---

## Step 3 — Build and load the serving image

From the **repo root** (`~/customer_churn_predition`):

```bash
docker build -f mlops/Dockerfile.kserve -t customer-churn-kserve:latest .
```

Load into kind. If `kind load` says *no nodes found for cluster "kind"*, pass your cluster name:

```bash
kind load docker-image customer-churn-kserve:latest --name mlops
```

(`kind get clusters` shows the name.)

---

## Step 4 — Deploy the InferenceService

Always run `kubectl apply` from the repo root (or use the full path). Applying from `~` fails with *path does not exist*.

**Option A — kind + MLflow on same VM (recommended for lab):**

```bash
cd ~/customer_churn_predition
export MLFLOW_HOST_IP="$(ip -4 route show default | awk '{print $3}')"
envsubst < mlops/kserve/inference-service-kind.yaml | kubectl apply -f -
```

**Option B — edit** `mlops/kserve/inference-service.yaml` and set `MLFLOW_TRACKING_URI` to a URL pods can resolve, then:

```bash
kubectl apply -f mlops/kserve/inference-service.yaml
```

Watch rollout:

```bash
kubectl get inferenceservice customer-churn-xgboost
kubectl get pods -l serving.kserve.io/inferenceservice=customer-churn-xgboost
kubectl describe inferenceservice customer-churn-xgboost
```

On Standard mode without Istio, **`IngressReady False`** and **`VirtualServiceCRDNotFound`** are expected. Check the **predictor pod** is Running and logs are clean:

```bash
kubectl logs -l serving.kserve.io/inferenceservice=customer-churn-xgboost \
  -c kserve-container --tail=100
```

---

## Step 5 — Call the model

**Port-forward** (typical on kind without ingress):

```bash
kubectl get svc | grep churn
kubectl port-forward svc/customer-churn-xgboost-predictor-default 8080:80
```

In another shell:

```bash
curl -s http://127.0.0.1:8080/v1/models/customer-churn-xgboost:predict \
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
  }' | python3 -m json.tool
```

Response fields match FastAPI: `churn_probability`, `churn_prediction`, `risk_band`, `threshold`.

---

## Promote Staging → Production

1. Set alias **Production** in MLflow UI or API ([mlflow_guide.md](mlflow_guide.md)).
2. Update the deployment env and roll:

```bash
kubectl set env deployment/$(kubectl get deploy -o name | grep customer-churn-xgboost | head -1 | cut -d/ -f2) \
  MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Production
```

Or edit the YAML and re-apply.

---

## Local smoke test (no Kubernetes)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Staging
pip install -r mlops/requirements-kserve.txt
python -m ml.serving.kserve_model
```

Then POST to `http://127.0.0.1:8080` with the same JSON as above.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `no matches for kind InferenceService` | KServe CRDs not installed | Step 1; verify `kubectl get crd inferenceservices.serving.kserve.io` |
| `path mlops/kserve/... does not exist` | Wrong working directory | `cd ~/customer_churn_predition` before `kubectl apply` |
| `no nodes found for cluster "kind"` | Cluster not named `kind` | `kind load ... --name mlops` |
| `MinimumReplicasUnavailable` | Image missing in cluster | `kind load docker-image customer-churn-kserve:latest --name mlops` |
| Pod crash on startup | `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in pod | Use host IP / `inference-service-kind.yaml` |
| `Unable to download artifacts` | S3 or artifact root | Align MLflow `--default-artifact-root` with pod credentials |
| `READY False`, ingress warnings only | No Istio on Standard mode | Ignore ingress; fix predictor pod + port-forward |
| `quick_install.sh` 404 | Script moved | Use release URL in Step 1 |

**Tests** (V2 tensor parsing):

```bash
pytest tests/test_kserve_handlers.py -q
```

---

## Related docs

- [mlflow_guide.md](mlflow_guide.md) — tracking, registry, FastAPI serving
- [mlops_lifecycle_guide.md](mlops_lifecycle_guide.md) — broader MLOps flow
- `mlops/docker-compose.yml` — Docker FastAPI serving (non-K8s)
