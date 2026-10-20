# KServe integration — serve the MLflow Staging model

End-to-end guide for this repo: **kind** on Ubuntu EC2, **KServe 0.20 (Standard mode)**, **MLflow** on the host, custom image `mlops/Dockerfile.kserve`.

The predictor uses the same pipeline as FastAPI (`ml/serving/predictor.py`) via **`MLFLOW_MODEL_URI`** (e.g. `models:/customer-churn-xgboost@Staging`).

---

## Recommended approach (start here)

You already use **MLflow registry** and (typically) **S3** for artifacts (DVC/training on EC2). The best lab setup that matches production patterns is **Path A**. Use **Path B** only for a quick demo without wiring AWS into Kubernetes.

| Path | When to use | Registry `@Staging` | S3 creds in pod | Manifest |
| --- | --- | --- | --- | --- |
| **A — Recommended** | Your EC2 + kind + MLflow + `s3://` artifacts | Yes | Yes (Kubernetes Secret) | `inference-service-kind.yaml` |
| **B — Lab shortcut** | Fastest demo; model file baked in Docker image | No (local joblib) | No | `inference-service-kind.yaml` + `KSERVE_USE_LOCAL_JOBLIB` |
| **C — Disk artifacts only** | MLflow `--default-artifact-root ./mlartifacts` (no S3) | Yes | No | `inference-service-kind.yaml` (drop `envFrom`) |

**Why Path A is best for you:** Training and registry stay the source of truth; promoting **Staging → Production** only changes `MLFLOW_MODEL_URI`. The pod needs the **same AWS identity** as the VM because `load_model(models:/...)` downloads from **S3**, not just the MLflow HTTP API (this caused your `NoCredentialsError`).

**Documentation map**

| Topic | Doc |
| --- | --- |
| Train, register, Staging alias | [mlflow_guide.md](mlflow_guide.md) |
| Docker/kind/KServe install (Steps 0–1) | This file, below |
| **Single checklist (Path A)** | [Quick runbook — Path A](#quick-runbook--path-a-registry--s3-on-ec2) |
| FastAPI serving (no K8s) | [mlflow_guide.md](mlflow_guide.md) Part 2, `api/main.py` |

### Architecture (Path A)

```text
  EC2 VM
  ├── MLflow (:5000) ──metadata──► sqlite + registry aliases
  ├── S3 bucket ◄──artifacts──► registered model versions
  └── kind cluster "mlops"
        └── Pod (hostNetwork)
              ├── MLFLOW_TRACKING_URI=http://127.0.0.1:5000
              ├── MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Staging
              └── AWS_* from Secret mlflow-aws-credentials
                    └── mlflow.sklearn.load_model → predict :8080
```

`hostNetwork: true` lets the pod use **`127.0.0.1:5000`** for MLflow on the same VM (simplest on kind).

---

## Quick runbook — Path A (registry + S3 on EC2)

Run from a clean shell on the VM. Assumes Steps **0** and **1** are already done (kind `mlops`, KServe controller Running).

```bash
# 1) Repo up to date (stash local edits to inference-service.yaml if needed)
cd ~/customer_churn_predition
git stash push -m local-isvc mlops/kserve/inference-service.yaml 2>/dev/null || true
git pull origin dev

# 2) MLflow running on the host (~/mlflow-setup) — see Step 2 below for full command
curl -s -o /dev/null -w "mlflow HTTP %{http_code}\n" http://127.0.0.1:5000/

# 3) Model in registry with Staging alias
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m ml.training.train   # skip if Staging already exists

# 4) AWS secret for the pod (same account/keys as training; file is gitignored)
cp mlops/kserve/aws-credentials-secret.example.yaml mlops/kserve/aws-credentials-secret.yaml
nano mlops/kserve/aws-credentials-secret.yaml   # fill AWS_ACCESS_KEY_ID, SECRET, REGION
kubectl apply -f mlops/kserve/aws-credentials-secret.yaml

# 5) Serving image
docker build -f mlops/Dockerfile.kserve -t customer-churn-kserve:latest .
kind load docker-image customer-churn-kserve:latest --name mlops

# 6) Deploy (hostNetwork + registry + envFrom secret)
kubectl delete inferenceservice customer-churn-xgboost --ignore-not-found
kubectl apply -f mlops/kserve/inference-service-kind.yaml

# 7) Wait and verify
kubectl get pods -l serving.kserve.io/inferenceservice=customer-churn-xgboost -w
kubectl logs -l serving.kserve.io/inferenceservice=customer-churn-xgboost -c kserve-container --tail=30
# expect: "Model loaded successfully"

# 8) Predict (port-forward)
kubectl port-forward svc/customer-churn-xgboost-predictor-default 8080:80
# second shell: curl POST ... see Step 5
```

**Promote to Production:** set alias in MLflow UI, then `kubectl set env deployment/... MLFLOW_MODEL_URI=models:/customer-churn-xgboost@Production` ([details](#promote-staging--production)).

### Quick runbook — Path B (lab shortcut, no S3 in pod)

1. `python -m ml.training.train` then `docker build ...` (joblib copied into image).
2. Edit `mlops/kserve/inference-service-kind.yaml`: comment out entire `envFrom`, uncomment `KSERVE_USE_LOCAL_JOBLIB=true`, comment `MLFLOW_MODEL_URI`.
3. `kind load` + `kubectl apply -f mlops/kserve/inference-service-kind.yaml`.

Good for learning KServe; **not** a substitute for Path A when you care about registry promotion.

---

## What lives in the repo

| Path | Role |
| --- | --- |
| `ml/serving/kserve_model.py` | KServe `ModelServer` entrypoint |
| `ml/serving/kserve_handlers.py` | V1 `instances` / V2 tensor request parsing |
| `mlops/Dockerfile.kserve` | Serving image (port **8080**) |
| `mlops/requirements-kserve.txt` | `kserve` Python package for the image |
| `mlops/kserve/inference-service.yaml` | Generic `InferenceService` (custom MLflow URI) |
| `mlops/kserve/inference-service-kind.yaml` | **Recommended** EC2/kind manifest (`hostNetwork`, AWS secret) |
| `mlops/kserve/aws-credentials-secret.example.yaml` | Template → `aws-credentials-secret.yaml` (gitignored) |

## Prerequisites (after Step 0)

1. **kind** cluster **`mlops`** with context `kind-mlops` and node `mlops-control-plane` **Ready**.
2. **MLflow** tracking server on HTTP(S), model registered with alias **Staging** ([mlflow_guide.md](mlflow_guide.md)).

---

## Step 0 — Install Docker, kubectl, and kind (Ubuntu EC2)

These steps follow the official [kind quick start](https://kind.sigs.k8s.io/docs/user/quick-start/). Run them once on a fresh Ubuntu EC2 (or similar) VM before KServe.

### 0.1 — Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker

sudo usermod -aG docker "$USER"
newgrp docker
```

Verify:

```bash
docker --version
docker ps
```

### 0.2 — kubectl

```bash
sudo snap install kubectl --classic
kubectl version --client
```

### 0.3 — kind

Downloads the binary for **x86_64** or **ARM64**:

```bash
ARCH=$(uname -m)

case "$ARCH" in
  x86_64) KIND_ARCH=amd64 ;;
  aarch64) KIND_ARCH=arm64 ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

curl -Lo kind "https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-${KIND_ARCH}"
chmod +x kind
sudo mv kind /usr/local/bin/kind

kind version
```

### 0.4 — Create the cluster

```bash
kind create cluster --name mlops --wait 5m
```

kind pulls a Kubernetes node image and creates a single-node cluster by default.

### 0.5 — Verify the cluster

```bash
kubectl cluster-info --context kind-mlops
kubectl get nodes -o wide
kubectl get pods -A
kind get clusters
```

Expected: one node (e.g. `mlops-control-plane`) in **Ready** status.

### 0.6 — Smoke test (optional nginx deployment)

```bash
kubectl create deployment nginx --image=nginx:stable
kubectl rollout status deployment/nginx
kubectl get deployments
kubectl get pods
```

Confirm context before continuing to KServe:

```bash
kubectl config current-context    # kind-mlops
kubectl config use-context kind-mlops   # if needed
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

On the VM (separate venv under `~/mlflow-setup`), start the server so **pods on the Docker bridge can connect**.

Do **not** type a literal `...` at the end of the command — MLflow will fail with *unexpected extra argument (...)*. Copy the full block below (same as [mlflow_guide.md](mlflow_guide.md)):

```bash
cd ~/mlflow-setup
source .venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --allowed-hosts '*'
```

Background (survive SSH disconnect):

```bash
cd ~/mlflow-setup && source .venv/bin/activate
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 --workers 1 --allowed-hosts '*' \
  > mlflow-server.log 2>&1 &
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

### S3 artifacts and `NoCredentialsError` in the serving pod

If the MLflow server uses **`--default-artifact-root s3://...`**, `mlflow.sklearn.load_model(models:/...)` downloads weights from **S3**, not only from the tracking API. The VM may have `~/.aws/credentials`, but the **Kubernetes pod does not**, which produces:

`botocore.exceptions.NoCredentialsError: Unable to locate credentials`

**Fix A — pass AWS creds into the pod (registry + S3):**

```bash
cp mlops/kserve/aws-credentials-secret.example.yaml mlops/kserve/aws-credentials-secret.yaml
# edit keys (do not commit this file)
kubectl apply -f mlops/kserve/aws-credentials-secret.yaml
kubectl apply -f mlops/kserve/inference-service-kind.yaml
```

**Fix B — lab only, joblib in the image (no S3 in pod):**

After `python -m ml.training.train`, rebuild the image, then in `inference-service-kind.yaml` remove/comment `envFrom`, set `KSERVE_USE_LOCAL_JOBLIB=true`, and comment out `MLFLOW_MODEL_URI`.

**Fix C — local MLflow artifacts on disk** (`--default-artifact-root ./mlartifacts`): pods still pull via the tracking server; S3 creds are not needed unless the registered version points at `s3://`.

---

## Step 3 — Build and load the serving image

From the **repo root** (`~/customer_churn_predition`):

```bash
docker build -f mlops/Dockerfile.kserve -t customer-churn-kserve:latest .
```

Load into kind (**always pass your cluster name** if it is not the default `kind`):

```bash
kind get clusters
kind load docker-image customer-churn-kserve:latest --name mlops
```

If you omit `--name mlops`, you get *no nodes found for cluster "kind"* when the cluster is named `mlops`.

---

## Step 4 — Deploy the InferenceService

Always run `kubectl apply` from the repo root (or use the full path). Applying from `~` fails with *path does not exist*.

**Sync the repo** (needed for `inference-service-kind.yaml` on older clones):

```bash
cd ~/customer_churn_predition
git pull origin dev
test -f mlops/kserve/inference-service-kind.yaml && echo OK || echo "missing — git pull dev"
```

**Recommended — kind + MLflow on same VM (Path A):**

Apply the AWS secret first (Step 2 S3 section), then:

```bash
cd ~/customer_churn_predition
kubectl apply -f mlops/kserve/inference-service-kind.yaml
```

Uses `hostNetwork: true` and `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.

**Alternative — edit** `mlops/kserve/inference-service.yaml` without `hostNetwork` and set `MLFLOW_TRACKING_URI` to the kind gateway (e.g. `http://172.18.0.1:5000`), then:

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
| `unexpected extra argument (...)` | Typed `...` literally | Use full `mlflow server` block in Step 2 |
| `inference-service-kind.yaml: No such file` | Old clone | `git pull origin dev` |
| `envsubst: command not found` | Package missing | `sudo apt-get install -y gettext-base` |
| `CrashLoopBackOff`, exit **137**, `connection refused` on :8080 | Liveness killed container during slow MLflow load, or wrong MLflow IP | Rebuild image after `kserve_model.py` fix; use `startupProbe`; verify MLflow URL from a pod (below) |
| `ModelLoadFailed` | Pod cannot reach MLflow or artifacts | Fix `MLFLOW_TRACKING_URI`; read `kubectl logs … -c kserve-container` |
| `NoCredentialsError` | Artifacts on S3, no AWS env in pod | `mlflow-aws-credentials` Secret + `envFrom` (see Step 2) or `KSERVE_USE_LOCAL_JOBLIB` |

**Find MLflow URL from inside the cluster** (do not guess `172.31.32.1`):

```bash
export MLFLOW_HOST_IP=$(docker inspect mlops-control-plane --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}')
echo "MLFLOW_HOST_IP=$MLFLOW_HOST_IP"

kubectl run mlflow-curl --rm -it --restart=Never --image=curlimages/curl -- \
  curl -v --connect-timeout 5 "http://${MLFLOW_HOST_IP}:5000/"
```

If that fails, use **`inference-service-kind.yaml` with `hostNetwork: true`** so the pod can use `http://127.0.0.1:5000` for MLflow on the same VM (simplest kind + host MLflow setup).

If the pod exits with **code 1** within a few seconds, the Python process crashed during model load — always check:

```bash
kubectl logs -l serving.kserve.io/inferenceservice=customer-churn-xgboost -c kserve-container --tail=80
kubectl logs -l serving.kserve.io/inferenceservice=customer-churn-xgboost -c kserve-container --previous
```

`connection refused` on :8080 is normal **until** `load()` finishes; KServe opens HTTP only after load. A quick exit 1 means load threw (MLflow down, wrong URI, missing Staging model, artifact/S3 error).

Use whichever MLflow URL works, then re-apply and **rebuild/reload** the image after code changes:

```bash
docker build -f mlops/Dockerfile.kserve -t customer-churn-kserve:latest .
kind load docker-image customer-churn-kserve:latest --name mlops
kubectl rollout restart deployment -l serving.kserve.io/inferenceservice=customer-churn-xgboost
kubectl logs -f -l serving.kserve.io/inferenceservice=customer-churn-xgboost -c kserve-container --tail=50
```

Delete stale ReplicaSets if two predictor pods fight each other after env changes:

```bash
kubectl get pods -l serving.kserve.io/inferenceservice=customer-churn-xgboost
```

**Tests** (V2 tensor parsing):

```bash
pytest tests/test_kserve_handlers.py -q
```

---

## Related docs

- [mlflow_guide.md](mlflow_guide.md) — tracking, registry, FastAPI serving
- [mlops_lifecycle_guide.md](mlops_lifecycle_guide.md) — broader MLOps flow
- `mlops/docker-compose.yml` — Docker FastAPI serving (non-K8s)
