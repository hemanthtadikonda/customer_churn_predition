from data_preparation.clean import clean_frame
from data_preparation.prepare import prepare_dataset
from data_acquisition.synthetic_source import generate_operational_frame
import pandas as pd


def test_prepare_does_not_modify_input(tmp_path):
    frame = generate_operational_frame(n_records=30, random_seed=12)
    raw_path = tmp_path / "customers.csv"
    frame.to_csv(raw_path, index=False)
    original = raw_path.read_bytes()
    config = {
        "preparation": {
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
        }
    }
    lineage = prepare_dataset(raw_path, config=config)
    assert raw_path.read_bytes() == original
    assert (tmp_path / "interim" / "customers.cleaned.csv").exists()
    assert (tmp_path / "processed" / "customers.processed.csv").exists()
    assert lineage["n_rows_in"] == 30


def test_clean_drops_exact_duplicates():
    frame = generate_operational_frame(n_records=10, random_seed=1)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    cleaned = clean_frame(doubled)
    assert len(cleaned) == len(frame)
