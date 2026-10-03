# DVC runbook — training data versioning

This is the operating guide for data versioning on this project. Another engineer should be able to clone the repo, pull the training snapshot from S3, and retrain without guessing which CSV was used.

Official DVC docs: [https://dvc.org/doc](https://dvc.org/doc).

---

## 1. What we version (and what we do not)

| Artifact | Tool | Why |
| --- | --- | --- |
| Training table `data/raw/telco_churn.csv` | DVC + S3 | This is the **only** file `ml.training.train` reads (`configs/config.yaml` → `data.raw_path`). Models must be auditable against these bytes. |
| Pointer `data/raw/telco_churn.csv.dvc` | Git | Tiny file: md5, size, relative path. That hash **is** the dataset version id. |
| S3 remote URL | **Local only** (`.dvc/config.local`) | Each machine or CI job adds the remote at runtime. Bucket URLs are not committed. |
| Train stage `dvc.yaml` | Git | Reproducible training command, deps, params, model outputs. |
| `data/raw/api/`, `data/raw/database/`, `data/synthetic/` | Local only (gitignored) | Data-foundation extracts. They are **not** the training contract today. Do not `dvc add` the whole `data/` tree. |
| The CSV bytes themselves | S3 via `dvc push` | Git is the wrong store for large, changing tables. |

**Why only one dataset?** A model is trained on one table (then split in code). Versioning every landing-zone snapshot would mix “we pulled CRM pages” with “we fitted XGBoost.” Industry practice is: **version the freeze used for training** (and later, version processed features the same way if train starts reading those). Extra extracts stay in the lake with their own metadata until training is rewired.

**Why metadata in Git and data in S3?**

- Git gives review, history, and PRs on **what changed** (pointer hash, pipeline, code).
- S3 holds **content-addressed blobs**. Same md5 → one object; a new CSV → new hash → new object. That is cheaper and safer than committing 700 KB–GB files to Git.
- A teammate runs `git pull` then `dvc pull` and gets **the same bytes** the pointer describes, without re-running the generator.

---

## 2. Architecture

```text
  python -m ml.data.generate
           │
           ▼
  data/raw/telco_churn.csv          ← actual rows (gitignored)
           │
           ├── dvc add  →  telco_churn.csv.dvc  (Git)
           └── dvc push →  S3 (remote you configure locally)
           │
           ▼
  dvc repro  (stage: train in dvc.yaml)
           │
           ▼
  artifacts/models/*.joblib         ← optional second DVC push after train
```

Do **not** put `data/raw/telco_churn.csv` in `dvc.yaml` as `outs:` while a `.dvc` sidecar exists. DVC will error with “overlaps with an output of stage …”. Generate with Python, freeze with `dvc add`, train with `dvc repro`.

---

## 3. What is already in this repo

| Item | Value |
| --- | --- |
| Tracked training file | `data/raw/telco_churn.csv` |
| Pointer | `data/raw/telco_churn.csv.dvc` |
| Pipeline | `dvc.yaml` stage `train` |
| S3 remote | **Not in Git.** Add it on each machine (section 4). |

`.dvc/config` in the repo has no remote URL. `--local` writes to `.dvc/config.local`, which is gitignored.

---

## 4. Configure the S3 remote (every machine / CI)

Do this from the **repository root** after `pip install -r requirements.txt` (includes `dvc[s3]`).

### 4.1 AWS credentials

DVC uses the same chain as boto3. Pick one:

**EC2 (preferred):** attach an instance profile with `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` on your bucket.

**Laptop or any VM:**

```bash
aws configure
# or:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

Never commit keys. Do not put them in `.dvc/config`.

Create the bucket in the AWS console or CLI if it does not exist (`Block public access` on).

### 4.2 Add the remote locally (do not commit this)

Replace the bucket and prefix with yours.

```bash
dvc remote add -d --local churnstore s3://YOUR-BUCKET/YOUR-PREFIX
```

| Flag | Meaning |
| --- | --- |
| `-d` | Make this the default remote for `dvc push` / `dvc pull` |
| `--local` | Write `.dvc/config.local` only (gitignored). Teammates configure their own bucket or the same bucket without changing Git. |

Example:

```bash
dvc remote add -d --local churnstore s3://my-mlops-bucket/customer-churn/dvc
```

If a remote named `churnstore` already exists on this machine:

```bash
dvc remote modify --local churnstore url s3://YOUR-BUCKET/YOUR-PREFIX
dvc remote default churnstore
```

### 4.3 Verify

```bash
dvc remote list
# expected:  churnstore    s3://YOUR-BUCKET/YOUR-PREFIX    (default)

aws s3 ls s3://YOUR-BUCKET/YOUR-PREFIX/
```

If `dvc push` later says `Unable to locate credentials`, fix AWS auth and retry. Do not `dvc remote add` without `-d --local` into committed `.dvc/config`.

---

## 5. Create, freeze, and push the training dataset

```bash
python3 -m ml.data.generate
ls -l data/raw/telco_churn.csv
# expect ~5000 rows; path is relative, not /data/raw

dvc add data/raw/telco_churn.csv
dvc push
```

Git after a successful add (do not add the CSV):

```bash
git add data/raw/telco_churn.csv.dvc data/raw/.gitignore
git add .dvc/.gitignore .dvcignore .gitignore dvc.yaml
git status   # telco_churn.csv and .dvc/config.local must stay out of Git
```

`.gitignore` at repo root ignores `data/raw/**` **except** `.gitkeep`, `data/raw/.gitignore`, and `*.dvc`. That is the correct fix for “`.dvc` file is git-ignored.” Do **not** stop ignoring all of `data/raw/` or you will accidentally commit `api/` and `database/` snapshots.

There is **no** file named `git-ignored`. DVC means “this path matches Git ignore rules.” Check with:

```bash
git check-ignore -v data/raw/telco_churn.csv
git check-ignore -v data/raw/telco_churn.csv.dvc
# CSV should be ignored; .dvc should NOT be ignored
```

---

## 6. Train with the frozen file

```bash
dvc pull                    # if the CSV is not on disk
dvc repro                   # runs train; writes dvc.lock
dvc push                    # uploads new model outs if the train stage produced them
```

`dvc.lock` belongs in Git once `dvc repro` has succeeded. It pins code + data hash + params to that run.

Keep `params.yaml` aligned with `configs/config.yaml` for model params DVC watches. Training still **reads** `configs/config.yaml`.

---

## 7. New engineer / new EC2 (consume, do not regenerate)

```bash
git clone <repo>
cd customer_churn_predition
git checkout <branch>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc remote add -d --local churnstore s3://YOUR-BUCKET/YOUR-PREFIX
dvc pull
ls data/raw/telco_churn.csv
python3 -m ml.training.train     # or: dvc repro
```

Do **not** run `ml.data.generate` unless you intend to create a **new** dataset version (new md5, new `dvc add`, new commit, `dvc push`).

---

## 8. New dataset version (when you mean to change training data)

```bash
python3 -m ml.data.generate      # or copy a new extract onto data/raw/telco_churn.csv
dvc add data/raw/telco_churn.csv # updates md5 in the .dvc file
dvc push
git add data/raw/telco_churn.csv.dvc
git commit -m "data: freeze telco_churn snapshot <md5 prefix>"
dvc repro
dvc push
```

Old S3 objects remain (content-addressed). Git history of the `.dvc` file is the changelog of which snapshot was current.

---

## 9. Git vs S3 vs local cache

| Commit to Git | Push with DVC (S3) | Never commit |
| --- | --- | --- |
| `data/raw/telco_churn.csv.dvc` | CSV blob (md5 in cache) | `data/raw/telco_churn.csv` |
| `dvc.yaml`, `dvc.lock`, `params.yaml` | Model outs after `dvc repro` | `.dvc/cache/`, `.dvc/tmp/`, `.dvc/config.local` |
| `data/raw/.gitignore` | | `data/raw/api/`, `data/raw/database/` |

Do not upload folders to S3 with the AWS console as a second source of truth. Use `dvc push` so hashes stay aligned with Git pointers.

---

## 10. Review of a typical first DVC session (practices)

What was **correct**:

- Generate the training file with `python3 -m ml.data.generate`.
- Track **only** `telco_churn.csv`, not the whole `data/` tree.
- Put the pointer in Git; put bytes in S3 with `dvc push`.
- That pointer (md5) is how you identify the dataset used for training.

What to **avoid**:

| Action | Why it is wrong | Do this instead |
| --- | --- | --- |
| `dvc add` while `dvc.yaml` `collect` listed the same `outs` | DVC forbids two owners of one path | Freeze with `dvc add`; keep `dvc.yaml` as **train** only |
| `rm dvc.yaml` | Drops the reproducible train graph | Restore `dvc.yaml` from Git |
| Un-ignore all of `data/raw/` | API/DB extracts become commitable | Exception only for `*.dvc` and `data/raw/.gitignore` |
| Searching for a file `git-ignored` | That filename does not exist | Read `.gitignore`; `git check-ignore -v <path>` |
| `ls /data/raw` | That is an absolute OS path | `ls data/raw` from the repo root |
| Commit the S3 URL in `.dvc/config` | Couples the repo to one account/bucket | `dvc remote add -d --local …` (section 4) |
| `git add data/raw/api` or `database/` | Not the training freeze | Leave gitignored |
| `dvc push` without AWS credentials | `Unable to locate credentials` | Instance role or `aws configure`, then retry |

---

## 11. Identifying the dataset that trained a model

1. **Path** — always `data/raw/telco_churn.csv` unless `data.raw_path` changed.
2. **Pointer** — `data/raw/telco_churn.csv.dvc` → `md5:` field. That hash is the version.
3. **Object store** — `dvc push` stores that md5 under the S3 prefix. `dvc pull` restores those bytes.
4. **After train** — `dvc.lock` records the data hash the train stage consumed, if you ran `dvc repro`.

`artifacts/models/metadata.json` currently stores split sizes, not the md5. Until that is added, **Git `.dvc` + `dvc.lock` are the audit trail**.

---

## 12. DVC capabilities (interview and operations)

Use these as talking points; they match this repo:

| Feature | What it does here |
| --- | --- |
| Data versioning | md5 of `telco_churn.csv` in Git; bytes in S3 |
| Separation of code and data | PRs review `.dvc` diffs, not 5000-row CSVs |
| Remote storage | `dvc remote` + `dvc push` / `dvc pull` |
| Pipelines | `dvc.yaml` `train` reruns only when deps/params change (`dvc repro`) |
| Metrics | JSON metrics in `dvc.yaml` (`cache: false` so they stay easy to diff in Git) |
| Cache | `.dvc/cache` is local; never commit it |
| Reproducibility | Same pointer + same code → same training input |

Advantages vs “commit CSV to Git” or “S3 upload by hand”:

- Reviewable history of **which** data, not a giant binary in `git log`.
- Deduplicated storage (same file hashed once).
- `dvc pull` on CI/EC2 without copying files through GitHub.
- Pipeline cache: skip train if data and code did not change.

Limits (say this in interviews too): DVC is not a feature store, not an experiment tracker (this repo uses MLflow for runs), and not a model registry. It versions files and pipelines.

---

## 13. AWS IAM (minimum)

On **your** bucket and prefix (`s3://YOUR-BUCKET/YOUR-PREFIX/`):

- `s3:ListBucket` on `arn:aws:s3:::YOUR-BUCKET`
- `s3:GetObject` and `s3:PutObject` on `arn:aws:s3:::YOUR-BUCKET/YOUR-PREFIX/*`

Block public access on the bucket. Prefer an EC2 instance profile over long-lived access keys on the VM.

---

## 14. Checklist

- [ ] `dvc remote add -d --local churnstore s3://YOUR-BUCKET/YOUR-PREFIX` has been run on this machine
- [ ] `dvc remote list` shows that URL as default; `.dvc/config.local` is **not** in Git
- [ ] `git check-ignore` ignores the CSV, not the `.dvc` file
- [ ] Git has `telco_churn.csv.dvc`, `data/raw/.gitignore`, `dvc.yaml`
- [ ] Git does **not** have the CSV, `api/`, `database/`, AWS keys, or a committed remote URL
- [ ] `dvc pull` restores `data/raw/telco_churn.csv` on a clean machine (after local remote + credentials)
- [ ] `dvc repro` trains against that file
- [ ] A new snapshot means new md5, new commit, `dvc push` — not silent overwrite without updating the pointer

---

## Related

- `data/README.md` — lake layout and immutability
- `docs/decisions.md` — why only the training CSV is DVC-tracked
- `docs/mlops_lifecycle_guide.md` — Session D
- [DVC user guide](https://dvc.org/doc/user-guide)
