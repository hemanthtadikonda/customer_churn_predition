"""Orchestrate raw -> cleaned (interim) -> processed copies.

Raw files are never overwritten. This module is a framework only: it does not
train models or create ML features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_acquisition.common import load_acquisition_config, project_path, sha256_file, utc_now_iso, write_json
from data_preparation.clean import clean_frame
from data_preparation.transform import transform_frame


def prepare_dataset(input_path: Path, *, config: dict | None = None) -> dict:
    """Create interim (cleaned) and processed copies of a validated extract."""
    cfg = config or load_acquisition_config()
    frame = _read_table(input_path)
    cleaned = clean_frame(frame)
    processed = transform_frame(cleaned)

    stem = input_path.stem
    interim_path = project_path(cfg["preparation"]["interim_dir"]) / f"{stem}.cleaned.csv"
    processed_path = project_path(cfg["preparation"]["processed_dir"]) / f"{stem}.processed.csv"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(interim_path, index=False)
    processed.to_csv(processed_path, index=False)

    lineage = {
        "raw_input": str(input_path),
        "raw_sha256": sha256_file(input_path) if input_path.exists() else None,
        "interim_path": str(interim_path),
        "processed_path": str(processed_path),
        "n_rows_in": int(len(frame)),
        "n_rows_cleaned": int(len(cleaned)),
        "n_rows_processed": int(len(processed)),
        "prepared_at_utc": utc_now_iso(),
        "note": "Raw file was not modified. Feature engineering is intentionally out of scope.",
    }
    write_json(processed_path.with_suffix(".lineage.json"), lineage)
    return lineage


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned and processed copies from a raw/validated file.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    lineage = prepare_dataset(Path(args.input))
    print(f"interim={lineage['interim_path']}")
    print(f"processed={lineage['processed_path']}")


if __name__ == "__main__":
    main()
