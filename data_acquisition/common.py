"""Shared utilities for the local data-foundation packages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data_acquisition" / "config.yaml"

SECRET_ENV_KEYS = {
    "POSTGRES_PASSWORD",
    "CHURN_API_TOKEN",
    "password",
    "token",
    "secret",
    "authorization",
}


class DataFoundationError(Exception):
    """Base error for the data foundation."""


class AcquisitionError(DataFoundationError):
    """Raised when a data source cannot be acquired."""


class CatalogPolicyError(AcquisitionError):
    """Raised when a catalog entry is missing, unapproved, or lacks provenance."""


def project_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path against the project root."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured-enough logs without emitting secrets."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not path.exists():
        raise AcquisitionError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise AcquisitionError(f"Expected a mapping in {path}")
    return payload


def load_acquisition_config(path: Path | None = None) -> dict[str, Any]:
    """Load acquisition configuration."""
    return load_yaml(path or DEFAULT_CONFIG_PATH)


def load_catalog(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the dataset catalog referenced by configuration."""
    cfg = config or load_acquisition_config()
    catalog_path = project_path(cfg["paths"]["catalog"])
    return load_yaml(catalog_path)


def get_dataset_entry(
    dataset_id: str,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one catalog record, failing closed if policy checks fail."""
    catalog_doc = catalog or load_catalog()
    datasets = catalog_doc.get("datasets") or []
    match = next((item for item in datasets if item.get("id") == dataset_id), None)
    if match is None:
        known = [item.get("id") for item in datasets]
        raise CatalogPolicyError(f"Dataset '{dataset_id}' is not in the catalog. Known ids: {known}")
    return match


def require_approved_download(entry: dict[str, Any]) -> None:
    """Refuse to download datasets that lack approval, URL, or license notes."""
    dataset_id = entry.get("id", "<unknown>")
    if not entry.get("approved_for_automatic_download"):
        raise CatalogPolicyError(
            f"Dataset '{dataset_id}' is not approved for automatic download. "
            "Update data/sources/dataset_catalog.yaml after reviewing provenance and license."
        )
    if not entry.get("download_url"):
        raise CatalogPolicyError(f"Dataset '{dataset_id}' has no download_url.")
    license_text = str(entry.get("license") or "").strip()
    if not license_text or license_text.lower() in {"unknown", "unclear"}:
        raise CatalogPolicyError(
            f"Dataset '{dataset_id}' cannot be downloaded because its license is undocumented."
        )


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable without logging its value."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def redact(value: str) -> str:
    """Replace secret-looking values before they can be logged."""
    if not value:
        return value
    return "***REDACTED***"


def metadata_path_for(data_path: Path) -> Path:
    """Sidecar metadata path for an acquired file."""
    return data_path.with_name(data_path.name + ".metadata.json")


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    backoff_seconds: float,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """HTTP helper with timeouts, limited retries, and 429 handling."""
    logger = logging.getLogger(__name__)
    last_error: Exception | None = None
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            response = client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout_seconds,
            )
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", backoff_seconds * attempt))
                logger.warning("Rate limited on %s; sleeping %.1fs (attempt %s)", url, retry_after, attempt)
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                logger.warning("Server error %s on %s (attempt %s)", response.status_code, url, attempt)
                time.sleep(backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            logger.warning("Request failed for %s (attempt %s/%s): %s", url, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(backoff_seconds * attempt)
    raise AcquisitionError(f"HTTP request failed for {url}: {last_error}") from last_error
