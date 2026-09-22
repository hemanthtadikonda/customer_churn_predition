# Data sources

## Primary public dataset

**IBM Telco Customer Churn** (`ibm_telco_customer_churn`)

| Field | Value |
| --- | --- |
| Publisher | IBM (Cognos Analytics / Cloud Pak for Data sample) |
| Landing page | https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113 |
| Download | IBM GitHub raw CSV in `IBM/telco-customer-churn-on-icp4d` |
| License | Apache-2.0 on the code-pattern repo; sample-data license not published separately |
| Size | ~950 KB, 7,043 rows, 21 columns |
| Target | `Churn` (`Yes` / `No`) |
| Real customers? | No. Fictional California telco sample |

### Why it is primary

- Documented publisher and canonical GitHub copy (no Kaggle token required)
- Schema matches the existing ML contract already in this repository
- Small, interview-standard, and clearly fictional
- Acquisition is gated by `approved_for_automatic_download: true` in the catalog

Treat it as a **public sample**, never as production CDR data.

## Approved alternative

**UCI Iranian Churn** (`uci_iranian_churn`)

| Field | Value |
| --- | --- |
| Publisher | UCI Machine Learning Repository |
| DOI | 10.24432/C5JW3Z |
| License | CC BY 4.0 |
| Size | 128.6 KB, 3,150 rows, 13 features |
| Target | `Churn` (0/1) |
| Notes | 9-month aggregates, label after 12 months; no public customer id |

Prefer this id in `data_acquisition/config.yaml` when a Creative Commons license is required.

## Documented but not auto-downloaded

- **Orange Telecom / BigML split** — usage-based features; license/canonical URL unclear on mirrors
- **Cell2Cell Duke/Teradata case** — useful historically; calibration oversampled; no clear redistribution license

The acquisition client **refuses** these unless the catalog is updated after a legal/provenance review.

## Simulated operational sources

These are **not** public datasets.

1. **SQLite local database** — tables `customers`, `subscriptions`, `billing`, `support_interactions`, `customer_usage`. Seeded from the synthetic generator. Stands in for PostgreSQL.
2. **Mock HTTP API** — paginated `/v1/customers`. Header `X-Simulation: true`. Stands in for a vendor/CRM API. No fake internet brand is claimed.
3. **Synthetic snapshots** — `data/synthetic/2026-01` … `2026-03` with increasing `drift_factor`.

## How sources are selected at runtime

```yaml
# data_acquisition/config.yaml
csv:
  dataset_id: ibm_telco_customer_churn
database:
  driver: sqlite
api:
  mode: mock
```

Switching PostgreSQL later is a config + env-var change, not a redesign of extract queries.
