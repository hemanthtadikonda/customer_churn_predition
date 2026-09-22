"""Extract raw snapshots from the simulated customer database.

Current driver: SQLite (local simulation).
Production driver: PostgreSQL via environment variables, same query interface.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

import pandas as pd

from data_acquisition.common import (
    AcquisitionError,
    load_acquisition_config,
    project_path,
    read_env,
    sha256_file,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger(__name__)


class DatabaseConnection(Protocol):
    def cursor(self) -> Any: ...

    def close(self) -> None: ...


def extract_database_snapshots(
    *,
    config: dict[str, Any] | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Run configured queries and write immutable CSV snapshots under data/raw/database."""
    cfg = config or load_acquisition_config()
    db_cfg = cfg["database"]
    stamp = snapshot_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = project_path(db_cfg["snapshot_dir"]) / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    try:
        with connect(db_cfg) as connection:
            for table_name, query in db_cfg["queries"].items():
                LOGGER.info("Extracting table %s", table_name)
                frame = _read_sql(connection, query)
                output_path = snapshot_dir / f"{table_name}.csv"
                frame.to_csv(output_path, index=False)
                files.append(
                    {
                        "table": table_name,
                        "query": query,
                        "path": str(output_path),
                        "rows": int(len(frame)),
                        "sha256": sha256_file(output_path),
                    }
                )
    except sqlite3.Error as exc:
        raise AcquisitionError(f"Database is unavailable or query failed: {exc}") from exc
    except AcquisitionError:
        raise
    except Exception as exc:
        raise AcquisitionError(f"Database extraction failed: {exc}") from exc

    metadata = {
        "source_type": "database_snapshot",
        "classification": "local_simulation",
        "driver": db_cfg["driver"],
        "connection_label": connection_label(db_cfg),
        "acquired_at_utc": utc_now_iso(),
        "snapshot_dir": str(snapshot_dir),
        "files": files,
        "note": "LOCAL SIMULATION of a production customer database. Not a live billing system.",
    }
    write_json(snapshot_dir / "_metadata.json", metadata)
    LOGGER.info("Wrote %s database snapshots to %s", len(files), snapshot_dir)
    return metadata


def _read_sql(connection: Any, query: str) -> pd.DataFrame:
    """Load a SQL result without requiring SQLAlchemy."""
    cursor = connection.execute(query)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def connection_label(db_cfg: dict[str, Any]) -> str:
    """Human-readable connection string without secrets."""
    driver = db_cfg["driver"]
    if driver == "sqlite":
        return f"sqlite:///{project_path(db_cfg['sqlite_path'])}"
    pg = db_cfg["postgresql"]
    host = read_env(pg["host_env"], "localhost")
    port = read_env(pg["port_env"], "5432")
    database = read_env(pg["database_env"], "churn_ops")
    user = read_env(pg["user_env"], "churn_app")
    return f"postgresql://{user}@{host}:{port}/{database}"


@contextmanager
def connect(db_cfg: dict[str, Any]) -> Iterator[Any]:
    """Yield a DB-API connection for the configured driver."""
    driver = db_cfg["driver"]
    if driver == "sqlite":
        path = project_path(db_cfg["sqlite_path"])
        if not path.exists():
            raise AcquisitionError(
                f"SQLite simulation database not found at {path}. "
                "Run: python -m data_acquisition.database_init"
            )
        connection = sqlite3.connect(path, timeout=float(db_cfg["connect_timeout_seconds"]))
        try:
            yield connection
        finally:
            connection.close()
        return

    if driver == "postgresql":
        yield from _connect_postgres(db_cfg)
        return

    raise AcquisitionError(f"Unsupported database driver '{driver}'. Use sqlite or postgresql.")


def _connect_postgres(db_cfg: dict[str, Any]) -> Iterator[Any]:
    try:
        import psycopg
    except ImportError as exc:
        raise AcquisitionError(
            "PostgreSQL driver selected, but psycopg is not installed. "
            "Keep driver=sqlite locally, or install psycopg in production."
        ) from exc

    pg = db_cfg["postgresql"]
    password = read_env(pg["password_env"])
    if password is None:
        raise AcquisitionError(
            f"Environment variable {pg['password_env']} is required for PostgreSQL and must not be hardcoded."
        )
    conninfo = (
        f"host={read_env(pg['host_env'], 'localhost')} "
        f"port={read_env(pg['port_env'], '5432')} "
        f"dbname={read_env(pg['database_env'], 'churn_ops')} "
        f"user={read_env(pg['user_env'], 'churn_app')} "
        f"connect_timeout={int(db_cfg['connect_timeout_seconds'])}"
    )
    LOGGER.info("Connecting to PostgreSQL at %s", connection_label(db_cfg))
    connection = psycopg.connect(conninfo, password=password)
    try:
        yield connection
    finally:
        connection.close()


def _cli() -> None:
    import argparse

    from data_acquisition.common import setup_logging

    parser = argparse.ArgumentParser(description="Extract raw snapshots from the local simulation database.")
    parser.add_argument("--snapshot-id", default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        metadata = extract_database_snapshots(snapshot_id=args.snapshot_id)
    except AcquisitionError as exc:
        raise SystemExit(f"Database extraction failed: {exc}") from exc
    print(f"Extracted {len(metadata['files'])} tables to {metadata['snapshot_dir']}")


if __name__ == "__main__":
    _cli()
