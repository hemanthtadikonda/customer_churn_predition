# Experiment tracking with MLflow

DVC versions **which dataset and pipeline** produced a model. MLflow records **each training run**: hyperparameters, metrics, tags, and artifacts so you can **compare experiments** in a UI.

This guide is for a **Linux VM**: MLflow **server** in a **separate directory and venv**; this Git repo is only the **client**.

---

## 1. Why tracking is required

Without a tracker, each `python -m ml.training.train` overwrites `artifacts/metrics/metrics.json`. You cannot answer:

- Which `learning_rate` / `max_depth` gave the best `test_roc_auc`?
- Was that run trained on the same CSV DVC froze (`dataset_sha256`)?
- Did recall drop when we changed the threshold?

Spreadsheets and terminal paste do not scale. Experiment tracking is the audit log of **model development**, the same way Git is the audit log of **code**.

| Tool | Question it answers |
| --- | --- |
| Git | Which code and config? |
| DVC | Which training file bytes (`telco_churn.csv.dvc`)? |
| MLflow | How did this run perform vs the last ten? |

They complement each other. Do not replace DVC with MLflow or the reverse.

---

## 2. Solution on this project

```text
  ~/mlflow-setup/.venv          MLflow *server*
       sqlite:///mlflow.db      run metadata (params, metrics)
       ./mlartifacts            logged files / sklearn model
       :5000                    UI + tracking API

  ~/customer_churn_predition/.venv     this repo (*client*)
       MLFLOW_TRACKING_URI=http://127.0.0.1:5000
       python -m ml.training.train
```

- **Server** stores history. You start it once and leave it running.
- **Client** is training code. It does not need to live in the same folder as the server.
- Default without the env var: local `./mlruns` (file store). Fine for a laptop; **not** how you compare runs across SSH sessions on a shared VM.

SQLite is enough for one VM and learning. PostgreSQL plus S3 artifact root is the production pattern (noted at the end).

The open-source tracking server has **no login**. Do not open port 5000 to the world. Prefer `http://127.0.0.1:5000` from training on the same VM, and an SSH tunnel or a security-group rule limited to your IP for the UI.

---

## 3. Install the MLflow server (separate directory)

Use a path **outside** this Git repo (example: home directory). Do **not** install the server into `customer_churn_predition/.venv`.

```bash
# as the user who will run the server (prefer ubuntu, not root, if you can)
mkdir -p ~/mlflow-setup
cd ~/mlflow-setup

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlflow

mlflow --version
```

Initialize the SQLite backend:

```bash
cd ~/mlflow-setup
source .venv/bin/activate
mlflow db upgrade sqlite:///mlflow.db
```

Start the server (keep this terminal open, or use `nohup` below):

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

| Flag | Why |
| --- | --- |
| `--backend-store-uri` | Where run/param/metric rows live |
| `--default-artifact-root` | Where metrics JSON and logged models go |
| `--host 0.0.0.0` | Listen on all interfaces (needed if you browse via the VM public IP) |
| `--allowed-hosts '*'` | Recent MLflow rejects unknown `Host` headers otherwise |

**Survive SSH disconnect:**

```bash
cd ~/mlflow-setup
source .venv/bin/activate
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --allowed-hosts '*' \
  > mlflow-server.log 2>&1 &
```

**Check it:**

```bash
mlflow --version
sudo ss -tulpn | grep 5000
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000
```

UI:

- On the VM: `http://127.0.0.1:5000`
- From your laptop: `http://<VM-PUBLIC-IP>:5000` only if the security group allows TCP **5000** from your IP

**SSH tunnel** (no public 5000):

```bash
ssh -L 5000:127.0.0.1:5000 ubuntu@<VM-PUBLIC-IP>
```

Then open `http://127.0.0.1:5000` on the laptop.

Optional smoke test (server venv or project venv):

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python3 - <<'PY'
import mlflow
mlflow.set_experiment("demo-experiment")
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
print("ok")
PY
```

Confirm **demo-experiment** in the UI, then use the churn experiment from training.

---

## 4. Code changes in this repository (client)

Training already logged params/metrics to **local** `mlruns/`. It now:

1. Reads **`MLFLOW_TRACKING_URI`** first (`ml/training/train.py`).
2. If the URI is `http://` or `https://`, it talks to your server.
3. If unset, it still uses `configs/config.yaml` → `mlflow.tracking_uri` (`mlruns/`).
4. Logs **tags** `dataset_path`, `dataset_sha256` so a run can be tied to the DVC snapshot.
5. Logs **threshold**, split sizes, and validation/test metrics for comparison.

No change is required inside `~/mlflow-setup`. Do not copy this repo into that folder.

`.env.example` documents `MLFLOW_TRACKING_URI`. Training reads the **environment**, not `.env`, unless you `export` the variable (or `set -a; source .env`).

---

## 5. Point this project at the server and train

In a **second** terminal, project venv — server must already be listening:

```bash
cd ~/customer_churn_predition
source .venv/bin/activate

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo "$MLFLOW_TRACKING_URI"

# training CSV must exist (DVC pull or generate)
dvc pull   # if you use DVC + remote
# or: python3 -m ml.data.generate

python3 -m ml.training.train
# or: dvc repro
```

Persist for the shell session in `~/.bashrc` **only on the VM** if you want it always on:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Use **127.0.0.1** when client and server share the VM. The public IP is for your browser, not required for logging.

Unset to fall back to local files:

```bash
unset MLFLOW_TRACKING_URI
```

---

## 6. How to use the UI (compare experiments)

Open the UI → experiment **`customer-churn-xgboost`**.

Each train creates a run named `xgboost_churn`.

**Compare two hyperparameter choices**

1. Change `model.params` in `configs/config.yaml` (keep `params.yaml` in sync if you use `dvc repro`).
2. Train again with `MLFLOW_TRACKING_URI` set.
3. In the UI, select two runs → **Compare**.
4. Sort by `test_roc_auc` or `test_recall`.

**What to look at**

| Field | Meaning |
| --- | --- |
| Params (`max_depth`, `learning_rate`, …) | What you changed |
| `threshold` | Serving cutoff used when metrics were computed |
| `test_roc_auc` / `test_pr_auc` / `test_recall` | Quality on held-out test rows |
| `val_*` | Tuning signal; do not pick a model using test only if you iterate a lot |
| Tag `dataset_sha256` | Same hash as the training file; should match the DVC freeze if you did not regenerate the CSV |
| Artifacts | `metrics.json`, `feature_importance.json`, logged sklearn pipeline |

**Best-practice workflow**

1. Freeze data with DVC; do not silently regenerate CSV between comparison runs.
2. Change **one** thing per run (or log a tag `hypothesis=deeper_trees`).
3. Promote a model using **test** metrics plus the quality gate (`roc_auc >= 0.75`), not training loss.
4. Keep `artifacts/models/churn_pipeline.joblib` as the serving copy; MLflow is the **ledger**, the joblib is what the API loads today.

---

## 7. Best practices (this stage)

- **Separate venvs:** server vs project. Version-skew between client and server causes odd REST errors; keep both on current `mlflow` when you can.
- **Same-VM URI:** `http://127.0.0.1:5000`.
- **Do not commit** `mlruns/`, `mlflow.db`, or `~/mlflow-setup`.
- **Do not log secrets** as params.
- **SQLite:** one writer. One `mlflow server` process is enough on this VM.
- **Security group:** port 5000 only from your IP, or use an SSH tunnel and bind the server to `127.0.0.1` instead of `0.0.0.0`.
- **Production later:** PostgreSQL backend, S3 `--default-artifact-root s3://…`, and an authenticated reverse proxy. Not required for this learning VM.

---

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| Runs appear under `customer_churn_predition/mlruns` | `echo $MLFLOW_TRACKING_URI` is empty; export it, retrain |
| `Unable to connect` / connection refused | Server not running; `ss` / `curl` on 5000 |
| `Invalid Host header` | Restart server with `--allowed-hosts '*'` |
| UI empty after train | Wrong experiment name, or trained against file store |
| Runs search `INTERNAL_ERROR` | Filter references params/metrics this project does not log (e.g. `rmse`, `params.model`). Clear the filter or use `metrics.test_roc_auc > 0.75` and `params.model_family = "xgboost"` |
| `MLflow model logging skipped` / untrusted types | Retrain on code with `skops_trusted_types` in `ml/training/train.py`, or ignore if you only need metrics (joblib on disk is unchanged) |
| `Unable to locate credentials` | That is DVC/S3, not MLflow |
| Quality gate fails | Tracking still works; fix model/metrics first |

---

## Related

- `docs/dvc_setup.md` — dataset versioning
- `configs/config.yaml` — `mlflow.experiment_name`
- `ml/training/train.py` — client logging
- [MLflow tracking](https://mlflow.org/docs/latest/tracking.html)
