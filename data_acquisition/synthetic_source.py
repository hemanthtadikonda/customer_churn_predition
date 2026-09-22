"""Synthetic telecom operational data with documented, non-independent relationships.

This generator does not produce real customer data. Every row is simulated from
explicit business assumptions so later stages can test validation, drift, and
retraining without pretending a public CSV is a production warehouse.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_acquisition.common import (
    AcquisitionError,
    load_acquisition_config,
    project_path,
    sha256_file,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger(__name__)

SYNTHETIC_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "contract_type",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "data_usage_gb",
    "voice_usage_minutes",
    "support_calls",
    "complaints",
    "late_payments",
    "discount_percent",
    "last_login_days",
    "churn",
    "snapshot_period",
]

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

ASSUMPTIONS = {
    "tenure_reduces_churn": "Each additional tenure month lowers log-odds of churn.",
    "month_to_month_higher_churn": "Month-to-month contracts have higher churn than two-year contracts.",
    "support_and_complaints_increase_churn": "More support calls and complaints raise churn probability.",
    "late_payments_increase_churn": "Repeated late payments correlate with churn.",
    "high_charges_increase_churn": "Unusually high monthly charges raise churn probability.",
    "inactivity_increases_churn": "Longer last_login_days raises churn probability.",
    "usage_drives_charges": "Higher data and voice usage increase monthly_charges.",
    "total_charges_from_tenure": "total_charges approximates monthly_charges * tenure with noise.",
    "discounts_reduce_churn": "Larger discount_percent slightly lowers churn probability.",
    "drift_is_simulated": "Later snapshots increase charges, tickets, inactivity, and baseline churn.",
    "not_real_customers": "Rows are synthetic. Do not present them as production CRM extracts.",
}


def generate_operational_frame(
    n_records: int = 2000,
    random_seed: int = 42,
    drift_factor: float = 0.0,
    period: str = "2026-01",
) -> pd.DataFrame:
    """Generate one synthetic snapshot with correlated operational fields."""
    if n_records <= 0:
        raise AcquisitionError("number_of_records must be positive.")
    rng = np.random.default_rng(random_seed)

    age = np.clip(rng.normal(42, 12, n_records).round().astype(int), 18, 80)
    tenure_raw = rng.gamma(shape=2.1, scale=12.0, size=n_records)
    tenure_months = np.clip(tenure_raw.astype(int), 0, 72)

    # Longer-tenure customers are more likely to be on annual contracts.
    contract_p = np.column_stack(
        [
            np.clip(0.70 - 0.007 * tenure_months, 0.20, 0.80),
            np.full(n_records, 0.18),
            np.clip(0.12 + 0.007 * tenure_months, 0.08, 0.55),
        ]
    )
    contract_p = contract_p / contract_p.sum(axis=1, keepdims=True)
    contract_type = np.array([CONTRACT_TYPES[int(rng.choice(3, p=p))] for p in contract_p])

    payment_method = rng.choice(
        PAYMENT_METHODS,
        size=n_records,
        p=[0.34, 0.18, 0.24, 0.24],
    )

    data_usage_gb = np.clip(rng.lognormal(mean=2.4, sigma=0.55, size=n_records), 0.2, 80.0)
    voice_usage_minutes = np.clip(rng.gamma(shape=4.0, scale=80.0, size=n_records), 0.0, 2000.0)
    data_usage_gb = np.round(data_usage_gb * (1 + 0.12 * drift_factor), 2)
    voice_usage_minutes = np.round(voice_usage_minutes, 1)

    monthly_charges = np.round(
        18.0
        + 1.15 * data_usage_gb
        + 0.02 * voice_usage_minutes
        + rng.normal(0, 3.5, n_records)
        + 8.0 * drift_factor,
        2,
    )
    monthly_charges = np.clip(monthly_charges, 15.0, 140.0)
    tenure_for_total = np.maximum(tenure_months, 1)
    total_charges = np.round(monthly_charges * tenure_for_total * rng.uniform(0.90, 1.08, n_records), 2)
    total_charges = np.where(tenure_months == 0, 0.0, total_charges)

    complaints = rng.binomial(1, np.clip(0.08 + 0.12 * drift_factor, 0.02, 0.40), n_records)
    support_calls = rng.poisson(1.1 + 1.8 * complaints + 0.8 * drift_factor, n_records)
    support_calls = np.clip(support_calls, 0, 18)
    late_payments = rng.poisson(0.4 + 0.6 * (contract_type == "Month-to-month") + 0.4 * drift_factor, n_records)
    late_payments = np.clip(late_payments, 0, 12)
    discount_percent = np.where(
        contract_type == "Two year",
        rng.choice([0, 5, 10, 15], size=n_records, p=[0.25, 0.35, 0.25, 0.15]),
        rng.choice([0, 5, 10], size=n_records, p=[0.60, 0.30, 0.10]),
    ).astype(float)
    last_login_days = np.clip(
        rng.gamma(shape=1.6, scale=6.0, size=n_records) + 8.0 * drift_factor,
        0,
        90,
    ).round().astype(int)

    logit = (
        -2.15
        - 0.040 * tenure_months
        + 0.90 * (contract_type == "Month-to-month")
        - 0.65 * (contract_type == "Two year")
        + 0.22 * support_calls
        + 0.55 * complaints
        + 0.28 * late_payments
        + 0.35 * (monthly_charges > 90)
        + 0.45 * (last_login_days > 21)
        + 0.18 * (age < 25)
        - 0.02 * discount_percent
        + 0.85 * drift_factor
        + rng.normal(0, 0.15, n_records)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(size=n_records) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:06d}" for i in range(1, n_records + 1)],
            "age": age,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "data_usage_gb": data_usage_gb,
            "voice_usage_minutes": voice_usage_minutes,
            "support_calls": support_calls,
            "complaints": complaints,
            "late_payments": late_payments,
            "discount_percent": discount_percent,
            "last_login_days": last_login_days,
            "churn": churn,
            "snapshot_period": period,
        }
    )
    return frame[SYNTHETIC_COLUMNS]


def generate_snapshots(
    *,
    config: dict[str, Any] | None = None,
    number_of_records: int | None = None,
    random_seed: int | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    """Write one or more time-based synthetic snapshots under data/synthetic/."""
    cfg = config or load_acquisition_config()
    synth_cfg = cfg["synthetic"]
    n_records = number_of_records if number_of_records is not None else int(synth_cfg["number_of_records"])
    seed = random_seed if random_seed is not None else int(synth_cfg["random_seed"])
    fmt = (output_format or synth_cfg["output_format"]).lower()
    if fmt not in {"csv", "parquet"}:
        raise AcquisitionError("output_format must be csv or parquet.")

    root = project_path(synth_cfg["output_dir"])
    written: list[dict[str, Any]] = []
    for snapshot in synth_cfg["snapshots"]:
        period = str(snapshot["period"])
        drift = float(snapshot.get("drift_factor", 0.0))
        # Offset the seed per period so snapshots differ, but remain reproducible.
        period_seed = seed + int(period.replace("-", ""))
        frame = generate_operational_frame(
            n_records=n_records,
            random_seed=period_seed,
            drift_factor=drift,
            period=period,
        )
        out_dir = root / period
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / f"customers.{fmt}"
        _write_frame(frame, data_path, fmt)
        snapshot_meta = {
            "source_type": "synthetic_operational",
            "classification": "synthetic",
            "real_customer_data": False,
            "period": period,
            "drift_factor": drift,
            "random_seed": period_seed,
            "base_random_seed": seed,
            "n_records": int(len(frame)),
            "churn_rate": float(frame["churn"].mean()),
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "columns": list(frame.columns),
            "assumptions": ASSUMPTIONS,
            "acquired_at_utc": utc_now_iso(),
        }
        write_json(out_dir / "_metadata.json", snapshot_meta)
        written.append(snapshot_meta)
        LOGGER.info(
            "Wrote synthetic snapshot %s (%s rows, churn_rate=%.1f%%) to %s",
            period,
            len(frame),
            100 * snapshot_meta["churn_rate"],
            data_path,
        )

    summary = {
        "source_type": "synthetic_operational",
        "output_dir": str(root),
        "snapshots": written,
        "note": "Synthetic data only. Relationships are simulation assumptions, not causal claims.",
    }
    write_json(root / "_generation_summary.json", summary)
    return summary


def _write_frame(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "csv":
        frame.to_csv(path, index=False)
        return
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise AcquisitionError("Parquet output requires pyarrow or fastparquet.") from exc


def _cli() -> None:
    import argparse

    from data_acquisition.common import setup_logging

    parser = argparse.ArgumentParser(description="Generate synthetic telecom operational snapshots.")
    parser.add_argument("--number-of-records", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--output-format", choices=["csv", "parquet"], default=None)
    args = parser.parse_args()
    setup_logging()
    summary = generate_snapshots(
        number_of_records=args.number_of_records,
        random_seed=args.random_seed,
        output_format=args.output_format,
    )
    print(f"Generated {len(summary['snapshots'])} synthetic snapshots under {summary['output_dir']}")


if __name__ == "__main__":
    _cli()
