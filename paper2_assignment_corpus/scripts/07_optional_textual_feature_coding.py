#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper2_assignments.conflict import add_conflict_fields
from paper2_assignments.io_utils import read_table, standardize_result_columns
from paper2_assignments.scoring import add_combined_scores
from paper2_assignments.textual_features import compare_feature_prevalence, run_feature_coding


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional de-identified OpenAI textual-feature coding for Paper 2.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = standardize_result_columns(read_table(args.input))
    if "combined_ai_rate_percent" not in data.columns:
        data = add_combined_scores(data)
    if "high_impact_conflict" not in data.columns:
        data = add_conflict_fields(data)
    codes = run_feature_coding(data, output, model=args.model, max_files=args.max_files)
    comparisons = compare_feature_prevalence(codes)
    comparisons.to_csv(output / "textual_feature_group_comparisons.csv", index=False, encoding="utf-8-sig")
    print(f"Textual-feature outputs saved to {output.resolve()}")


if __name__ == "__main__":
    main()
