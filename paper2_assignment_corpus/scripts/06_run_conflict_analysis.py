#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2_assignments.conflict import add_conflict_fields, conflict_by_group, conflict_summary, detector_low_high_patterns, pairwise_detector_differences
from paper2_assignments.io_utils import read_table, standardize_result_columns
from paper2_assignments.scoring import add_combined_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broad and high-impact detector-conflict analyses for Paper 2.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = standardize_result_columns(read_table(args.input))
    if "combined_ai_rate_percent" not in data.columns:
        data = add_combined_scores(data)
    data = add_conflict_fields(data)
    tables = {
        "assignment_conflict_fields": data,
        "conflict_summary": conflict_summary(data),
        "pairwise_detector_differences": pairwise_detector_differences(data),
        "conflict_by_language": conflict_by_group(data, "language"),
        "conflict_by_assignment_type": conflict_by_group(data, "assignment_type"),
        "high_impact_patterns": detector_low_high_patterns(data),
    }
    with pd.ExcelWriter(output / "paper2_conflict_analysis.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            if name != "assignment_conflict_fields":
                table.to_excel(writer, sheet_name=name[:31], index=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"Conflict outputs saved to {output.resolve()}")


if __name__ == "__main__":
    main()
