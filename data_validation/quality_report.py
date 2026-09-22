"""Human-readable quality report wrapper around validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_validation.validate import validate_file


def render_quality_report(report: dict) -> str:
    """Format a validation report for logs or a terminal."""
    lines = [
        f"Status: {report['status']}",
        f"Schema: {report.get('schema')}",
        f"Rows: {report.get('n_rows')}  Columns: {report.get('n_columns')}",
        f"Input: {report.get('input_path', '<in-memory>')}",
        "",
        "Checks:",
    ]
    for check in report.get("checks", []):
        lines.append(f"  [{check['status']}] {check['name']}: {check['message']}")
    lines.append("")
    lines.append(report.get("note", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a data-quality report.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", default="synthetic_operational")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    report = validate_file(Path(args.input), args.schema)
    print(render_quality_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
