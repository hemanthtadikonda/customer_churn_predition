from pathlib import Path

from data_acquisition.api_source import extract_api_snapshots
from data_acquisition.synthetic_source import generate_operational_frame


def test_mock_api_extract_stores_raw_json(tmp_path: Path):
    records = generate_operational_frame(n_records=12, random_seed=2).to_dict(orient="records")
    config = {
        "synthetic": {"number_of_records": 12, "random_seed": 2},
        "api": {
            "mode": "mock",
            "snapshot_dir": str(tmp_path / "raw" / "api"),
            "timeout_seconds": 5,
            "max_retries": 1,
            "retry_backoff_seconds": 0.01,
            "page_size": 5,
            "rate_limit_sleep_seconds": 0,
            "mock": {"host": "127.0.0.1", "port": 0, "label": "LOCAL SIMULATION"},
            "public": {"base_url_env": "CHURN_API_BASE_URL", "token_env": "CHURN_API_TOKEN"},
        },
    }
    metadata = extract_api_snapshots(config=config, records=records)
    assert metadata["mode"] == "mock"
    assert metadata["row_count"] == 12
    assert metadata["classification"] == "local_simulation"
    assert len(metadata["pages"]) == 3
    snapshot = Path(metadata["snapshot_dir"])
    assert (snapshot / "page_001.json").exists()
    assert "LOCAL SIMULATION" in (snapshot / "page_001.json").read_text(encoding="utf-8")
