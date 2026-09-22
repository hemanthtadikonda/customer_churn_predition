"""Run one or more acquisition sources in a defined order."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

from data_acquisition.api_source import extract_api_snapshots
from data_acquisition.common import AcquisitionError, load_acquisition_config, setup_logging, utc_now_iso
from data_acquisition.csv_source import download_public_csv
from data_acquisition.database_init import initialize_sqlite_database
from data_acquisition.database_source import extract_database_snapshots
from data_acquisition.synthetic_source import generate_snapshots

LOGGER = logging.getLogger(__name__)

SourceFn = Callable[..., dict[str, Any] | Any]


def run_orchestrator(
    sources: list[str],
    *,
    init_database: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute selected acquisition jobs. Validation is a separate step."""
    cfg = config or load_acquisition_config()
    results: dict[str, Any] = {"started_at_utc": utc_now_iso(), "sources": {}}
    selected = [item.strip().lower() for item in sources if item.strip()]
    unknown = sorted(set(selected) - {"csv", "database", "api", "synthetic"})
    if unknown:
        raise AcquisitionError(f"Unknown sources: {unknown}. Use csv, database, api, synthetic.")

    if init_database or "database" in selected:
        if cfg["database"]["driver"] == "sqlite":
            initialize_sqlite_database(config=cfg)

    runners: dict[str, SourceFn] = {
        "csv": lambda: download_public_csv(config=cfg),
        "database": lambda: extract_database_snapshots(config=cfg),
        "api": lambda: extract_api_snapshots(config=cfg),
        "synthetic": lambda: generate_snapshots(config=cfg),
    }
    for name in selected:
        LOGGER.info("Starting acquisition source: %s", name)
        results["sources"][name] = runners[name]()
    results["finished_at_utc"] = utc_now_iso()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire data from configured sources.")
    parser.add_argument(
        "--sources",
        default="synthetic",
        help="Comma-separated list: csv,database,api,synthetic",
    )
    parser.add_argument(
        "--init-database",
        action="store_true",
        help="Create/reseed the local SQLite simulation before extraction.",
    )
    args = parser.parse_args()
    setup_logging()
    try:
        results = run_orchestrator(args.sources.split(","), init_database=args.init_database)
    except AcquisitionError as exc:
        raise SystemExit(f"Orchestration failed: {exc}") from exc
    print("Acquisition complete:")
    for name, payload in results["sources"].items():
        print(f"  - {name}: {payload.get('output_path') or payload.get('snapshot_dir') or payload.get('output_dir')}")


if __name__ == "__main__":
    main()
