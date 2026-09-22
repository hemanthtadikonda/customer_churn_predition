# Design decisions

## Dataset selection

**Decision:** Primary public dataset is IBM Telco Customer Churn from IBM's GitHub sample, not a random Kaggle mirror and not an undocumented URL.

**Why:** Publisher is IBM, the file is fictional, the schema is the industry interview standard, and it matches the ML contract already in this repo. License is documented with an explicit caveat (Apache-2.0 on the code pattern; no standalone data license). Automatic download is allowed only because that caveat is recorded in `dataset_catalog.yaml`.

**Alternative considered:** UCI Iranian Churn (CC BY 4.0) — cleaner license, weaker schema overlap, no customer id. Kept as `approved_for_automatic_download` alternative.

**Rejected for auto-download:** Orange/BigML mirrors and Cell2Cell Kaggle copies — provenance or license not clear enough.

## Raw vs synthetic vs simulated

**Decision:** Three physically separate landing zones (`data/raw`, `data/synthetic`, SQLite under `data/sources`).

**Why:** Interviewers should hear that we never relabel generated rows as "production CRM data."

## Database

**Decision:** SQLite now, PostgreSQL-shaped connector later (`driver` in config, secrets only from env).

**Why:** The project must run on a laptop. Init is a separate module from extract so schema ownership stays obvious.

## API

**Decision:** Local mock HTTP API, clearly labeled. No invented public "telco vendor."

**Why:** There is no reputable public API that serves customer-level churn records under terms we can rely on. Faking one would be worse than a mock.

## Validation does not fix data

**Decision:** PASS/WARN/FAIL reports only.

**Why:** Repairing raw files destroys lineage. Fixes belong in preparation, with a new file.

## Preparation is a framework only

**Decision:** No one-hot encoding, scaling, or model features in `data_preparation/`.

**Why:** Those steps must stay with the training/serving code to avoid train-serve skew.

## DVC / cloud / training not expanded here

**Decision:** Do not add new DVC remotes, Docker, Kubernetes, or training jobs in this phase.

**Why:** The request is a data foundation. Some of those files already exist from an earlier portfolio step; they are preserved, not extended.

---

## Conflicts with the pre-existing repository

Inspected before adding this foundation. Nothing was deleted.

| Existing item | Conflict | Resolution |
| --- | --- | --- |
| `ml/data/generate.py` writes **synthetic IBM-schema rows into `data/raw/telco_churn.csv`** | Violates "raw is immutable public extract" and "do not pretend synthetic data is real" | Left in place so DVC `collect` and current training tests still run. New generators write to `data/synthetic/` and labeled raw subfolders. Comment added on the old module. |
| `dvc.yaml` collect stage | Treats generated CSV as the official raw dataset | Unchanged this phase. Future work should point DVC at catalog-acquired or prepared data. |
| `ml/data/schema.py` | IBM Yes/No schema vs synthetic 0/1 operational schema | Both documented. Validation supports named schemas. Do not silently union them. |
| `data/raw/.gitkeep` + `data/processed/.gitkeep` | Layout was incomplete (no interim/synthetic/sources) | Extended folders; kept existing gitkeeps. |
| `README.md` Python 3.11 + ML quickstart | This phase asks for 3.12+ and data-first docs | CI bumped to 3.12. Root README documents both the new data foundation and the existing ML path. |
| Docker, compose, MLflow, FastAPI | Out of scope for this phase | Preserved, unused by the new data packages. |

No existing tests were rewritten except that they must still pass alongside the new ones.
