from data_acquisition.synthetic_source import SYNTHETIC_COLUMNS, generate_operational_frame, generate_snapshots


def test_synthetic_schema_and_types():
    frame = generate_operational_frame(n_records=200, random_seed=7, period="2026-01")
    assert list(frame.columns) == SYNTHETIC_COLUMNS
    assert frame["customer_id"].is_unique
    assert frame["churn"].isin([0, 1]).all()
    assert frame["contract_type"].isin(["Month-to-month", "One year", "Two year"]).all()
    assert (frame["age"].between(18, 80)).all()
    assert (frame["monthly_charges"] > 0).all()
    assert 0.05 < float(frame["churn"].mean()) < 0.55


def test_synthetic_reproducibility():
    a = generate_operational_frame(n_records=150, random_seed=99, drift_factor=0.1, period="2026-02")
    b = generate_operational_frame(n_records=150, random_seed=99, drift_factor=0.1, period="2026-02")
    c = generate_operational_frame(n_records=150, random_seed=100, drift_factor=0.1, period="2026-02")
    assert a.equals(b)
    assert not a.equals(c)


def test_synthetic_relationships_are_not_independent():
    frame = generate_operational_frame(n_records=1500, random_seed=3, period="2026-01")
    mtm = frame.loc[frame["contract_type"] == "Month-to-month", "churn"].mean()
    two_year = frame.loc[frame["contract_type"] == "Two year", "churn"].mean()
    assert mtm > two_year
    high_support = frame.loc[frame["support_calls"] >= 3, "churn"].mean()
    low_support = frame.loc[frame["support_calls"] <= 1, "churn"].mean()
    assert high_support >= low_support


def test_synthetic_snapshots_are_seeded(tmp_path):
    config = {
        "synthetic": {
            "number_of_records": 40,
            "random_seed": 11,
            "output_format": "csv",
            "output_dir": str(tmp_path),
            "snapshots": [
                {"period": "2026-01", "drift_factor": 0.0},
                {"period": "2026-02", "drift_factor": 0.2},
            ],
        }
    }
    summary = generate_snapshots(config=config)
    assert len(summary["snapshots"]) == 2
    first = tmp_path / "2026-01" / "customers.csv"
    second = tmp_path / "2026-02" / "customers.csv"
    assert first.exists() and second.exists()
    generate_snapshots(config=config)
    assert first.read_bytes() == (tmp_path / "2026-01" / "customers.csv").read_bytes()
    assert first.read_bytes() != second.read_bytes()
