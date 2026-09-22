"""Public CSV acquisition for approved catalog datasets."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from data_acquisition.common import (
    AcquisitionError,
    CatalogPolicyError,
    get_dataset_entry,
    load_acquisition_config,
    load_catalog,
    metadata_path_for,
    project_path,
    require_approved_download,
    request_with_retries,
    sha256_bytes,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def download_public_csv(
    dataset_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Download an approved public dataset into data/raw without modifying it.

    Raw files are immutable snapshots. Re-running the job skips an existing
    file unless overwrite is enabled.
    """
    cfg = config or load_acquisition_config()
    catalog = load_catalog(cfg)
    selected_id = dataset_id or cfg["csv"]["dataset_id"]
    entry = get_dataset_entry(selected_id, catalog)
    require_approved_download(entry)

    raw_dir = project_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / str(entry["output_filename"])
    should_overwrite = cfg["csv"]["overwrite"] if overwrite is None else overwrite

    if output_path.exists() and not should_overwrite:
        metadata = _existing_or_refresh_metadata(output_path, entry)
        LOGGER.info("Raw file already present at %s; skipping download.", output_path)
        return metadata

    headers = {"User-Agent": cfg["csv"]["user_agent"]}
    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True)
    try:
        LOGGER.info("Downloading %s from catalog URL %s", selected_id, entry["download_url"])
        response = request_with_retries(
            http_client,
            "GET",
            entry["download_url"],
            max_retries=int(cfg["csv"]["max_retries"]),
            backoff_seconds=float(cfg["csv"]["retry_backoff_seconds"]),
            timeout_seconds=float(cfg["csv"]["timeout_seconds"]),
            headers=headers,
        )
        payload = response.content
    finally:
        if owns_client:
            http_client.close()

    _validate_download_payload(payload, entry)
    output_path.write_bytes(payload)
    metadata = build_csv_metadata(output_path, payload, entry)
    write_json(metadata_path_for(output_path), metadata)
    LOGGER.info(
        "Stored immutable raw snapshot %s (%s bytes, sha256=%s)",
        output_path,
        metadata["bytes"],
        metadata["sha256"],
    )
    return metadata


def build_csv_metadata(path: Path, payload: bytes, entry: dict[str, Any]) -> dict[str, Any]:
    """Create acquisition metadata for a public CSV snapshot."""
    frame = _parse_csv(payload, entry)
    return {
        "source_type": "public_csv",
        "classification": entry.get("classification"),
        "real_customer_data": entry.get("real_customer_data"),
        "dataset_id": entry.get("id"),
        "dataset_name": entry.get("name"),
        "publisher": entry.get("publisher"),
        "license": entry.get("license"),
        "landing_page": entry.get("landing_page"),
        "download_url": entry.get("download_url"),
        "acquired_at_utc": utc_now_iso(),
        "output_path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "columns": list(frame.columns),
        "immutable": True,
        "note": "Original bytes were written without transformation.",
    }


def validate_csv_payload(payload: bytes, entry: dict[str, Any]) -> pd.DataFrame:
    """Public helper used by tests to check empty/corrupt downloads."""
    _validate_download_payload(payload, entry)
    return _parse_csv(payload, entry)


def _validate_download_payload(payload: bytes, entry: dict[str, Any]) -> None:
    if not payload or not payload.strip():
        raise AcquisitionError(f"Downloaded file for '{entry.get('id')}' is empty.")
    min_bytes = int(entry.get("min_bytes") or 1)
    if len(payload) < min_bytes:
        raise AcquisitionError(
            f"Downloaded file for '{entry.get('id')}' is {len(payload)} bytes, "
            f"below min_bytes={min_bytes}. The file may be corrupt or truncated."
        )
    expected = entry.get("checksum_sha256")
    if expected:
        digest = sha256_bytes(payload)
        if digest != expected:
            raise AcquisitionError(
                f"Checksum mismatch for '{entry.get('id')}': expected {expected}, got {digest}."
            )


def _parse_csv(payload: bytes, entry: dict[str, Any]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # pandas ParserError and encoding issues
        raise AcquisitionError(f"Downloaded file for '{entry.get('id')}' is not a readable CSV: {exc}") from exc
    if frame.empty:
        raise AcquisitionError(f"Downloaded CSV for '{entry.get('id')}' contains no rows.")
    expected_columns = entry.get("expected_columns")
    if expected_columns and int(frame.shape[1]) != int(expected_columns):
        raise AcquisitionError(
            f"Downloaded CSV for '{entry.get('id')}' has {frame.shape[1]} columns, "
            f"expected {expected_columns}."
        )
    expected_rows = entry.get("n_records")
    if expected_rows and int(len(frame)) != int(expected_rows):
        LOGGER.warning(
            "Row count for '%s' is %s, catalog lists %s.",
            entry.get("id"),
            len(frame),
            expected_rows,
        )
    return frame


def _existing_or_refresh_metadata(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    sidecar = metadata_path_for(path)
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    metadata = build_csv_metadata(path, payload, entry)
    write_json(sidecar, metadata)
    return metadata


def _cli() -> None:
    import argparse

    from data_acquisition.common import setup_logging

    parser = argparse.ArgumentParser(description="Download an approved public churn dataset.")
    parser.add_argument("--dataset-id", default=None, help="Catalog id. Defaults to config csv.dataset_id.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing raw file.")
    args = parser.parse_args()
    setup_logging()
    try:
        metadata = download_public_csv(args.dataset_id, overwrite=args.overwrite or None)
    except (AcquisitionError, CatalogPolicyError) as exc:
        raise SystemExit(f"CSV acquisition failed: {exc}") from exc
    print(f"Acquired {metadata['dataset_id']} -> {metadata['output_path']}")
    print(f"sha256={metadata['sha256']} rows={metadata['row_count']}")


if __name__ == "__main__":
    _cli()
