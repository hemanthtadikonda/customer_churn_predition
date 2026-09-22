# Data lineage

```text
Public Dataset ──────┐
                     │
Local Database ──────┼──→ Raw Data
                     │
Public/Mock API ─────┘
                           │
                           ▼
                     Validation
                           │
                           ▼
                       Processed
                           │
                           ▼
                   Future ML Pipeline
```

Synthetic snapshots follow the same validation → processed path but are stored under `data/synthetic/` so they cannot be mistaken for public extracts.

## Stage by stage

### 1. Sources

| Source | Code | What happens |
| --- | --- | --- |
| Public dataset | `csv_source.py` | HTTPS GET of a **catalog-approved** URL. Original bytes written to `data/raw/`. SHA-256 + publisher/license recorded. |
| Local database | `database_init.py` then `database_source.py` | SQLite is created as a **simulation**, then `SELECT *` snapshots are written under `data/raw/database/`. |
| Mock API | `api_source.py` | Paginated JSON is stored first (`page_001.json`, …). A convenience CSV is derived after the raw pages exist. |

No source overwrites another source's filename on purpose: public IBM CSV, database folders, and API folders are separate.

### 2. Raw data

- Immutable
- Gitignored
- Accompanied by metadata JSON
- May be public-sample, simulated, or synthetic — classification is in metadata

If validation fails, raw stays. We do not "repair" the extract in place.

### 3. Validation

`data_validation.validate` reads a copy of the table in memory and writes a report to `data/interim/validation_reports/`.

Statuses:

- **PASS** — contract held
- **WARN** — usable with caveats (for example IBM `TotalCharges` blanks)
- **FAIL** — do not promote to processed/training

### 4. Processed (via cleaned interim)

`data_preparation.prepare`:

1. Reads the input file (never deletes it)
2. Writes a cleaned copy to `data/interim/`
3. Writes a processed copy to `data/processed/`
4. Writes lineage JSON with the raw checksum

Cleaning is limited to trim/dedupe/type coercion. It is **not** feature engineering.

### 5. Future ML pipeline

Out of scope for this phase. Later jobs will read `data/processed/` (or a warehouse table equivalent), split train/val/test, engineer features, train, register, and serve.

The existing `ml/` package still reads `data/raw/telco_churn.csv` from the older generator. That path is legacy until the training stage is rewired; see `docs/decisions.md`.

## Identity of a dataset version

```text
version = source_type + catalog_or_snapshot_id + sha256
```

Example: `public_csv / ibm_telco_customer_churn / <sha256>`
