"""Synthetic telco churn data collection.

In production a data scientist would pull this from a warehouse, CRM, or
billing system. Here we generate an IBM Telco-style table with realistic
churn drivers so the rest of the MLOps pipeline can run offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import load_config, resolve_path
from ml.data.schema import TARGET_COL, validate_dataframe


def _choice(rng: np.random.Generator, options: list, size: int, p=None) -> np.ndarray:
    return rng.choice(options, size=size, p=p)


def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    gender = _choice(rng, ["Female", "Male"], n_samples)
    senior = rng.binomial(1, 0.16, n_samples)
    partner = _choice(rng, ["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = _choice(rng, ["Yes", "No"], n_samples, p=[0.30, 0.70])

    tenure = np.clip(rng.gamma(shape=2.2, scale=14.0, size=n_samples).astype(int), 0, 72)
    phone_service = _choice(rng, ["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        _choice(rng, ["Yes", "No"], n_samples, p=[0.42, 0.58]),
    )

    internet_service = _choice(
        rng,
        ["DSL", "Fiber optic", "No"],
        n_samples,
        p=[0.34, 0.44, 0.22],
    )

    def internet_addon(prob_yes: float) -> np.ndarray:
        raw = _choice(rng, ["Yes", "No"], n_samples, p=[prob_yes, 1 - prob_yes])
        return np.where(internet_service == "No", "No internet service", raw)

    online_security = internet_addon(0.29)
    online_backup = internet_addon(0.34)
    device_protection = internet_addon(0.34)
    tech_support = internet_addon(0.29)
    streaming_tv = internet_addon(0.38)
    streaming_movies = internet_addon(0.39)

    contract = _choice(
        rng,
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.24, 0.21],
    )
    paperless = _choice(rng, ["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment = _choice(
        rng,
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_samples,
        p=[0.34, 0.19, 0.23, 0.24],
    )

    base_charge = np.where(
        internet_service == "Fiber optic",
        75,
        np.where(internet_service == "DSL", 50, 20),
    )
    addon_yes = sum(
        (col == "Yes").astype(int)
        for col in [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]
    )
    monthly = np.round(
        base_charge
        + 8 * (phone_service == "Yes")
        + 5 * (multiple_lines == "Yes")
        + 7 * addon_yes
        + rng.normal(0, 4, n_samples),
        2,
    )
    monthly = np.clip(monthly, 18.25, 118.75)
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.92, 1.05, n_samples), 2)
    total = np.where(tenure == 0, 0.0, total)

    # Logit built from well-known telco churn drivers, not random labels.
    logit = (
        -1.55
        + 1.15 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
        + 0.70 * (internet_service == "Fiber optic")
        + 0.55 * (payment == "Electronic check")
        + 0.45 * ((tech_support == "No") & (internet_service != "No"))
        + 0.35 * ((online_security == "No") & (internet_service != "No"))
        + 0.35 * senior
        - 0.035 * tenure
        + 0.012 * (monthly - 65)
        - 0.25 * (partner == "Yes")
        - 0.20 * (dependents == "Yes")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.uniform(size=n_samples) < probability, "Yes", "No")

    frame = pd.DataFrame(
        {
            "customerID": [f"CUST{i:06d}" for i in range(1, n_samples + 1)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            TARGET_COL: churn,
        }
    )
    validate_dataframe(frame, require_target=True)
    return frame


def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telco churn data.")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    n_samples = args.n_samples or int(config["data"]["n_samples"])
    output = Path(args.output) if args.output else resolve_path(config["data"]["raw_path"])

    df = generate_churn_dataset(n_samples=n_samples, random_state=config["data"]["random_state"])
    save_dataset(df, output)
    churn_rate = (df[TARGET_COL] == "Yes").mean()
    print(f"Wrote {len(df)} rows to {output} (churn rate={churn_rate:.1%})")


if __name__ == "__main__":
    main()
