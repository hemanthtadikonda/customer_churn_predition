from pathlib import Path

import httpx
import pytest
import yaml

from data_acquisition.common import (
    AcquisitionError,
    CatalogPolicyError,
    get_dataset_entry,
    require_approved_download,
    sha256_bytes,
    sha256_file,
)
from data_acquisition.csv_source import download_public_csv, validate_csv_payload


def _entry(**overrides):
    base = {
        "id": "test_public",
        "name": "Test Public CSV",
        "classification": "public_sample",
        "real_customer_data": False,
        "approved_for_automatic_download": True,
        "publisher": "Test",
        "download_url": "https://example.test/churn.csv",
        "license": "CC0-1.0",
        "output_filename": "test_churn.csv",
        "min_bytes": 10,
        "expected_columns": 3,
        "n_records": 2,
        "checksum_sha256": None,
    }
    base.update(overrides)
    return base


def test_sha256_matches_known_bytes(tmp_path: Path):
    payload = b"hello"
    path = tmp_path / "raw.csv"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)
    assert sha256_bytes(b"hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_validate_csv_payload_rejects_empty():
    with pytest.raises(AcquisitionError, match="empty"):
        validate_csv_payload(b"   ", _entry())


def test_validate_csv_payload_rejects_wrong_shape():
    csv = b"a,b\n1,2\n"
    with pytest.raises(AcquisitionError, match="columns"):
        validate_csv_payload(csv, _entry(min_bytes=1, expected_columns=5))


def test_validate_csv_payload_rejects_checksum_mismatch():
    csv = b"a,b,c\n1,2,3\n4,5,6\n"
    with pytest.raises(AcquisitionError, match="Checksum mismatch"):
        validate_csv_payload(csv, _entry(checksum_sha256="deadbeef", min_bytes=1))


def test_catalog_policy_blocks_unapproved_download():
    with pytest.raises(CatalogPolicyError, match="not approved"):
        require_approved_download(_entry(approved_for_automatic_download=False))


def test_catalog_policy_blocks_unknown_license():
    with pytest.raises(CatalogPolicyError, match="license"):
        require_approved_download(_entry(license="unknown"))


def test_get_dataset_entry_unknown_id():
    catalog = {"datasets": [_entry()]}
    with pytest.raises(CatalogPolicyError, match="not in the catalog"):
        get_dataset_entry("missing", catalog)


def test_download_public_csv_writes_file_and_metadata(tmp_path: Path, monkeypatch):
    csv = b"customerID,tenure,Churn\n0001,3,No\n0002,12,Yes\n"
    catalog = {"datasets": [_entry(min_bytes=20, expected_columns=3, n_records=2)]}
    config = {
        "paths": {"catalog": "unused.yaml", "raw_dir": str(tmp_path / "raw")},
        "csv": {
            "dataset_id": "test_public",
            "timeout_seconds": 5,
            "max_retries": 1,
            "retry_backoff_seconds": 0.01,
            "user_agent": "test",
            "overwrite": False,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/churn.csv"
        return httpx.Response(200, content=csv)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    monkeypatch.setattr("data_acquisition.csv_source.load_catalog", lambda _cfg: catalog)
    metadata = download_public_csv("test_public", config=config, client=client)
    output = Path(metadata["output_path"])
    assert output.exists()
    assert metadata["row_count"] == 2
    assert metadata["sha256"] == sha256_bytes(csv)
    sidecar = output.with_name(output.name + ".metadata.json")
    assert sidecar.exists()
    assert "license" in sidecar.read_text(encoding="utf-8")

    # Second call without overwrite must not hit the network.
    def fail_handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not download again")

    metadata_again = download_public_csv(
        "test_public",
        config=config,
        client=httpx.Client(transport=httpx.MockTransport(fail_handler)),
    )
    assert metadata_again["sha256"] == metadata["sha256"]


def test_real_catalog_primary_is_approved():
    catalog = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "data" / "sources" / "dataset_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert catalog["primary_dataset_id"] == "ibm_telco_customer_churn"
    entry = get_dataset_entry("ibm_telco_customer_churn", catalog)
    require_approved_download(entry)
    assert entry["publisher"] == "IBM"
    assert "Apache-2.0" in entry["license"]


def test_download_fails_on_http_error(tmp_path: Path, monkeypatch):
    catalog = {"datasets": [_entry()]}
    config = {
        "paths": {"catalog": "unused.yaml", "raw_dir": str(tmp_path / "raw")},
        "csv": {
            "dataset_id": "test_public",
            "timeout_seconds": 1,
            "max_retries": 1,
            "retry_backoff_seconds": 0.01,
            "user_agent": "test",
            "overwrite": True,
        },
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"missing")

    monkeypatch.setattr("data_acquisition.csv_source.load_catalog", lambda _cfg: catalog)
    with pytest.raises(AcquisitionError, match="HTTP request failed"):
        download_public_csv(
            "test_public",
            config=config,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
