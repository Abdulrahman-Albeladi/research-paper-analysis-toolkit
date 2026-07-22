#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast

import pandas as pd

from paper1_survey.qualitative import build_validation_workbook


def parse_theme_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [item.strip() for item in str(value).split(";") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-validation workbook for coded open responses.")
    parser.add_argument("--coded-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-question", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.coded_csv)
    if "themes" in frame.columns:
        frame["themes"] = frame["themes"].map(parse_theme_list)
    build_validation_workbook(frame, args.output, args.n_per_question, args.seed)
    print(f"Saved validation workbook to {args.output}")


if __name__ == "__main__":
    main()
