"""Local mock operational API.

This is a LOCAL SIMULATION of an external vendor/CRM API. It is not a real
telecom service and must not be described as one. A public third-party API is
not used because no reputable public API exposes customer-level churn records
under terms that fit this project.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from data_acquisition.common import (
    AcquisitionError,
    load_acquisition_config,
    project_path,
    read_env,
    request_with_retries,
    sha256_bytes,
    utc_now_iso,
    write_json,
)
from data_acquisition.synthetic_source import generate_operational_frame

LOGGER = logging.getLogger(__name__)


class MockTelecomAPIHandler(BaseHTTPRequestHandler):
    """Serves paginated synthetic customer records. Labeled as a simulation."""

    records: list[dict[str, Any]] = []
    simulation_label = "LOCAL SIMULATION — not a real external telecom API"

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("mock-api " + format, *args)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"status": "ok", "mode": "mock", "label": self.simulation_label})
            return
        if parsed.path != "/v1/customers":
            self._json(404, {"error": "not_found", "label": self.simulation_label})
            return

        query = parse_qs(parsed.query)
        page = max(int(query.get("page", ["1"])[0]), 1)
        page_size = min(max(int(query.get("page_size", ["100"])[0]), 1), 500)
        start = (page - 1) * page_size
        end = start + page_size
        total = len(self.records)
        if page > 3 and page * page_size > total + page_size:
            # Keep pagination simple; unknown huge pages still return an empty list.
            pass
        payload = {
            "label": self.simulation_label,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": end < total,
            "data": self.records[start:end],
        }
        self._json(200, payload)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Simulation", "true")
        self.end_headers()
        self.wfile.write(body)


def start_mock_server(
    records: list[dict[str, Any]],
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, str]:
    """Start the in-process mock API. Port 0 binds an ephemeral port."""
    handler = type(
        "BoundMockHandler",
        (MockTelecomAPIHandler,),
        {"records": records, "simulation_label": MockTelecomAPIHandler.simulation_label},
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    base_url = f"http://{bound_host}:{bound_port}"
    LOGGER.info("%s listening at %s", MockTelecomAPIHandler.simulation_label, base_url)
    return server, base_url


def extract_api_snapshots(
    *,
    config: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
    base_url: str | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pull paginated API JSON and store raw responses before any transformation."""
    cfg = config or load_acquisition_config()
    api_cfg = cfg["api"]
    if api_cfg["mode"] == "public":
        return _extract_public_api(api_cfg, client=client, base_url=base_url)

    if api_cfg["mode"] != "mock":
        raise AcquisitionError(f"Unsupported API mode '{api_cfg['mode']}'. Use mock or public.")

    server = None
    owns_server = base_url is None
    try:
        if owns_server:
            payload = records or generate_operational_frame(
                n_records=int(cfg["synthetic"]["number_of_records"]),
                random_seed=int(cfg["synthetic"]["random_seed"]),
                period="api-mock",
            ).to_dict(orient="records")
            server, base_url = start_mock_server(
                payload,
                host=str(api_cfg["mock"]["host"]),
                port=0,
            )
        assert base_url is not None
        return _paginate_and_store(api_cfg, base_url, client=client, mode="mock")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


def _extract_public_api(
    api_cfg: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    resolved = base_url or read_env(api_cfg["public"]["base_url_env"])
    if not resolved:
        raise AcquisitionError(
            "API mode is public, but CHURN_API_BASE_URL is not set. "
            "Do not hardcode endpoints; keep mode=mock unless a documented public API is configured."
        )
    token = read_env(api_cfg["public"]["token_env"])
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return _paginate_and_store(api_cfg, resolved, client=client, mode="public", headers=headers)


def _paginate_and_store(
    api_cfg: dict[str, Any],
    base_url: str,
    *,
    client: httpx.Client | None,
    mode: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    snapshot_dir = project_path(api_cfg["snapshot_dir"]) / utc_now_iso().replace(":", "")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http_client = client or httpx.Client(follow_redirects=True)
    pages: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    try:
        page = 1
        while True:
            url = f"{base_url.rstrip('/')}/v1/customers"
            LOGGER.info("Fetching API page %s from %s", page, url)
            response = request_with_retries(
                http_client,
                "GET",
                url,
                max_retries=int(api_cfg["max_retries"]),
                backoff_seconds=float(api_cfg["retry_backoff_seconds"]),
                timeout_seconds=float(api_cfg["timeout_seconds"]),
                headers=headers,
                params={"page": page, "page_size": int(api_cfg["page_size"])},
            )
            raw_path = snapshot_dir / f"page_{page:03d}.json"
            raw_path.write_bytes(response.content)
            pages.append({"page": page, "path": str(raw_path), "sha256": sha256_bytes(response.content)})
            body = response.json()
            rows = body.get("data") or []
            all_rows.extend(rows)
            if not body.get("has_more"):
                break
            page += 1
            if page > 100:
                raise AcquisitionError("API pagination exceeded 100 pages; aborting to avoid a runaway loop.")
    finally:
        if owns_client:
            http_client.close()

    frame = pd.DataFrame(all_rows)
    combined_path = snapshot_dir / "customers.csv"
    frame.to_csv(combined_path, index=False)
    metadata = {
        "source_type": "api_snapshot",
        "classification": "local_simulation" if mode == "mock" else "external_api",
        "mode": mode,
        "label": api_cfg["mock"]["label"] if mode == "mock" else "configured public API",
        "base_url": base_url,
        "acquired_at_utc": utc_now_iso(),
        "snapshot_dir": str(snapshot_dir),
        "pages": pages,
        "row_count": int(len(frame)),
        "note": "Raw JSON pages were stored before CSV combination. Mock mode is not a real vendor API.",
    }
    write_json(snapshot_dir / "_metadata.json", metadata)
    LOGGER.info("Stored %s API pages (%s rows) under %s", len(pages), len(frame), snapshot_dir)
    return metadata


def _cli() -> None:
    import argparse

    from data_acquisition.common import setup_logging

    parser = argparse.ArgumentParser(description="Extract snapshots from the local mock operational API.")
    args = parser.parse_args()
    del args
    setup_logging()
    try:
        metadata = extract_api_snapshots()
    except AcquisitionError as exc:
        raise SystemExit(f"API acquisition failed: {exc}") from exc
    print(f"Acquired API snapshot with {metadata['row_count']} rows at {metadata['snapshot_dir']}")


if __name__ == "__main__":
    _cli()
