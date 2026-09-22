# Data dictionary

Two contracts exist on purpose:

1. **IBM Telco public sample** — 21-column tutorial schema (`Churn` as Yes/No)
2. **Synthetic / simulated operational schema** — CRM-like fields used by SQLite, the mock API, and `data/synthetic/`

Do not mix them in one training table without an explicit mapping job.

Training eligibility below is for a **future** model. This phase does not train.

---

## Target: churn

| | IBM public sample | Synthetic / ops simulation |
| --- | --- | --- |
| Field | `Churn` | `churn` |
| Type | string | integer |
| Values | `Yes`, `No` | `1`, `0` |
| Nullable | no | no |
| Business meaning | Customer left during the observation window | Simulated binary exit label from a documented logit |
| Use for training | **Yes, as the label only** | **Yes, as the label only** |
| Leakage risk | **High if** you ever add IBM's extra fields `Churn Score`, `Churn Reason`, or `CLTV`. Those are outcomes or model scores, not pre-churn features. The 21-column file used here does not include them. | Label is generated from the same features; that is acceptable for a simulator but **overstates** real-world predictability. |

Never use the target as a feature. Never train on post-event fields (reason codes, win-back flags).

---

## IBM Telco public sample

| Field | Description | Type | Source | Range / values | Nullable | Business meaning | Train? | Leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| customerID | Stable customer key | string | IBM sample | opaque id | no | Join key | no (identifier) | none if dropped before fit |
| gender | Recorded gender | string | IBM sample | Female, Male | no | Demographic | optional; fairness review | proxy risk |
| SeniorCitizen | Senior flag | int | IBM sample | 0, 1 | no | Demographic | yes, with care | proxy risk |
| Partner | Has partner | string | IBM sample | Yes, No | no | Household | yes | low |
| Dependents | Has dependents | string | IBM sample | Yes, No | no | Household | yes | low |
| tenure | Months with the brand | int | IBM sample | 0–72 | no | Relationship length | yes | survivors bias in production |
| PhoneService | Phone subscription | string | IBM sample | Yes, No | no | Product | yes | low |
| MultipleLines | Extra lines | string | IBM sample | Yes, No, No phone service | no | Product | yes | must stay consistent with PhoneService |
| InternetService | Access type | string | IBM sample | DSL, Fiber optic, No | no | Product | yes | low |
| OnlineSecurity | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | consistency with InternetService |
| OnlineBackup | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | same |
| DeviceProtection | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | same |
| TechSupport | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | same |
| StreamingTV | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | same |
| StreamingMovies | Add-on | string | IBM sample | Yes, No, No internet service | no | Product | yes | same |
| Contract | Commitment term | string | IBM sample | Month-to-month, One year, Two year | no | Commercial | yes | strong driver; not leakage |
| PaperlessBilling | E-bill | string | IBM sample | Yes, No | no | Billing channel | yes | low |
| PaymentMethod | How they pay | string | IBM sample | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | no | Billing | yes | low |
| MonthlyCharges | Recurring bill | float | IBM sample | > 0 | no | Revenue | yes | low |
| TotalCharges | Cumulative bill | numeric / blank | IBM sample | blank when tenure=0 | **yes (blanks)** | Historical spend | yes | partly determined by tenure × monthly; still a standard feature, not a post-churn label |

---

## Synthetic operational fields

Simulation assumptions are in `data_acquisition/synthetic_source.py` (`ASSUMPTIONS`). They are **not** causal claims about real telecoms.

| Field | Description | Type | Source | Range / values | Nullable | Business meaning | Train? | Leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| customer_id | Synthetic key | string | generator / SQLite / mock API | `CUST######` | no | Join key | no | none if dropped |
| age | Simulated age | int | generator | 18–80 | no | Demographic | optional | proxy risk |
| tenure_months | Relationship length | int | generator | 0–72 | no | Loyalty proxy | yes | survivors bias |
| contract_type | Commitment | string | generator | Month-to-month, One year, Two year | no | Commercial | yes | assumed churn driver |
| payment_method | How they pay | string | generator | same four IBM methods | no | Billing | yes | low |
| monthly_charges | Recurring bill | float | generator | ~15–140 | no | Driven by usage | yes | low |
| total_charges | Cumulative bill | float | generator | ≥ 0 | no | `monthly ≈ × tenure` | yes | collinear with tenure |
| data_usage_gb | Data volume | float | generator | ≥ 0 | no | Usage | yes | low |
| voice_usage_minutes | Voice volume | float | generator | ≥ 0 | no | Usage | yes | low |
| support_calls | Care contacts | int | generator | ≥ 0 | no | Friction | yes | assumed driver |
| complaints | Complaint flag | int | generator | 0, 1 | no | Friction | yes | assumed driver |
| late_payments | Late-pay count | int | generator | ≥ 0 | no | Credit friction | yes | assumed driver |
| discount_percent | Offer discount | float | generator | 0–100 | no | Retention offer | **careful** | can be a treatment, not a pre-treatment feature |
| last_login_days | Recency of activity | int | generator | 0–90 | no | Engagement | yes | assumed driver |
| snapshot_period | Partition month | string | generator | `YYYY-MM` | no | Time axis for drift | no as a raw feature unless used as time | time leakage if mixed randomly |
| churn | Simulated label | int | generator | 0, 1 | no | See target section | label only | generated from features |

---

## Simulated database tables

Same grain as the synthetic customer row, normalized:

- `customers(customer_id, age, tenure_months, last_login_days, churn)`
- `subscriptions(subscription_id, customer_id, contract_type, payment_method, discount_percent)`
- `billing(billing_id, customer_id, monthly_charges, total_charges, late_payments)`
- `support_interactions(interaction_id, customer_id, support_calls, complaints)`
- `customer_usage(usage_id, customer_id, data_usage_gb, voice_usage_minutes)`

Extracts land in `data/raw/database/<snapshot>/` as CSV. Joining them back is a preparation/warehouse step, not an acquisition step.
