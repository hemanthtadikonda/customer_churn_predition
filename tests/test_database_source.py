from pathlib import Path

import pandas as pd
import pytest

from data_acquisition.common import AcquisitionError
from data_acquisition.database_init import initialize_sqlite_database
from data_acquisition.database_source import extract_database_snapshots


def _db_config(tmp_path: Path) -> dict:
    db_path = tmp_path / "local_ops.sqlite"
    snapshot_dir = tmp_path / "raw" / "database"
    return {
        "database": {
            "driver": "sqlite",
            "sqlite_path": str(db_path),
            "snapshot_dir": str(snapshot_dir),
            "connect_timeout_seconds": 5,
            "postgresql": {
                "host_env": "POSTGRES_HOST",
                "port_env": "POSTGRES_PORT",
                "database_env": "POSTGRES_DB",
                "user_env": "POSTGRES_USER",
                "password_env": "POSTGRES_PASSWORD",
            },
            "queries": {
                "customers": "SELECT * FROM customers",
                "subscriptions": "SELECT * FROM subscriptions",
                "billing": "SELECT * FROM billing",
                "support_interactions": "SELECT * FROM support_interactions",
                "customer_usage": "SELECT * FROM customer_usage",
            },
        },
        "database_init": {"n_customers": 25, "random_seed": 5},
        "synthetic": {},
    }


def test_database_init_and_extract(tmp_path: Path):
    config = _db_config(tmp_path)
    db_path = initialize_sqlite_database(config=config)
    assert db_path.exists()
    metadata = extract_database_snapshots(config=config, snapshot_id="testsnap")
    assert metadata["classification"] == "local_simulation"
    assert metadata["driver"] == "sqlite"
    assert len(metadata["files"]) == 5
    customers = pd.read_csv(Path(metadata["snapshot_dir"]) / "customers.csv")
    assert len(customers) == 25
    assert customers["customer_id"].is_unique
    assert (tmp_path / "raw" / "database" / "testsnap" / "_metadata.json").exists()


def test_database_extract_fails_when_missing(tmp_path: Path):
    config = _db_config(tmp_path)
    with pytest.raises(AcquisitionError, match="not found"):
        extract_database_snapshots(config=config)
