"""Local SQLite initialization for the simulated operational database.

This module creates schema and seed data. It does not extract snapshots.
Extraction lives in database_source.py.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from data_acquisition.common import AcquisitionError, load_acquisition_config, project_path, write_json
from data_acquisition.synthetic_source import generate_operational_frame

LOGGER = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS customer_usage;
DROP TABLE IF EXISTS support_interactions;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    tenure_months INTEGER NOT NULL,
    last_login_days INTEGER NOT NULL,
    churn INTEGER NOT NULL CHECK (churn IN (0, 1))
);

CREATE TABLE subscriptions (
    subscription_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    contract_type TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    discount_percent REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE billing (
    billing_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges REAL NOT NULL,
    late_payments INTEGER NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_interactions (
    interaction_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    support_calls INTEGER NOT NULL,
    complaints INTEGER NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE customer_usage (
    usage_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    data_usage_gb REAL NOT NULL,
    voice_usage_minutes REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""


def initialize_sqlite_database(
    db_path: Path | None = None,
    *,
    n_customers: int | None = None,
    random_seed: int | None = None,
    config: dict | None = None,
) -> Path:
    """Create a local SQLite database that simulates production CRM/billing tables."""
    cfg = config or load_acquisition_config()
    path = db_path or project_path(cfg["database"]["sqlite_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    n_customers = n_customers if n_customers is not None else int(cfg["database_init"]["n_customers"])
    random_seed = random_seed if random_seed is not None else int(cfg["database_init"]["random_seed"])

    frame = generate_operational_frame(
        n_records=n_customers,
        random_seed=random_seed,
        drift_factor=0.0,
        period="seed",
    )
    if path.exists():
        path.unlink()

    LOGGER.info("Initializing LOCAL SIMULATION SQLite database at %s", path)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        _insert_normalized_tables(connection, frame)
        connection.commit()

    sidecar = path.with_suffix(path.suffix + ".init.json")
    write_json(
        sidecar,
        {
            "source_type": "sqlite_local_simulation",
            "path": str(path),
            "n_customers": n_customers,
            "random_seed": random_seed,
            "note": "This file is a local stand-in for PostgreSQL/Aurora/RDS. It is not a production database.",
        },
    )
    LOGGER.info("Seeded %s customers into %s", n_customers, path)
    return path


def _insert_normalized_tables(connection: sqlite3.Connection, frame: pd.DataFrame) -> None:
    customers = [
        (row.customer_id, int(row.age), int(row.tenure_months), int(row.last_login_days), int(row.churn))
        for row in frame.itertuples(index=False)
    ]
    subscriptions = [
        (f"SUB{i:06d}", row.customer_id, row.contract_type, row.payment_method, float(row.discount_percent))
        for i, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    billing = [
        (f"BIL{i:06d}", row.customer_id, float(row.monthly_charges), float(row.total_charges), int(row.late_payments))
        for i, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    support = [
        (f"SUP{i:06d}", row.customer_id, int(row.support_calls), int(row.complaints))
        for i, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    usage = [
        (f"USE{i:06d}", row.customer_id, float(row.data_usage_gb), float(row.voice_usage_minutes))
        for i, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    connection.executemany(
        "INSERT INTO customers (customer_id, age, tenure_months, last_login_days, churn) VALUES (?, ?, ?, ?, ?)",
        customers,
    )
    connection.executemany(
        "INSERT INTO subscriptions (subscription_id, customer_id, contract_type, payment_method, discount_percent) VALUES (?, ?, ?, ?, ?)",
        subscriptions,
    )
    connection.executemany(
        "INSERT INTO billing (billing_id, customer_id, monthly_charges, total_charges, late_payments) VALUES (?, ?, ?, ?, ?)",
        billing,
    )
    connection.executemany(
        "INSERT INTO support_interactions (interaction_id, customer_id, support_calls, complaints) VALUES (?, ?, ?, ?)",
        support,
    )
    connection.executemany(
        "INSERT INTO customer_usage (usage_id, customer_id, data_usage_gb, voice_usage_minutes) VALUES (?, ?, ?, ?)",
        usage,
    )


def _cli() -> None:
    import argparse

    from data_acquisition.common import setup_logging

    parser = argparse.ArgumentParser(description="Initialize the local simulated operations database.")
    parser.add_argument("--n-customers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        path = initialize_sqlite_database(n_customers=args.n_customers, random_seed=args.random_seed)
    except AcquisitionError as exc:
        raise SystemExit(f"Database initialization failed: {exc}") from exc
    print(f"Initialized local simulation database: {path}")


if __name__ == "__main__":
    _cli()
