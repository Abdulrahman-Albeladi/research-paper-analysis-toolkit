#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper1_survey.pipeline import run_main_analysis_from_path
from paper1_survey.schema import DEFAULT_SCHEMA, SurveySchema


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 1 descriptive, psychometric, and ordinal analyses.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema")
    args = parser.parse_args()
    schema = SurveySchema.from_json(args.schema) if args.schema else DEFAULT_SCHEMA
    run_main_analysis_from_path(args.input, args.output, schema)
    print(f"Paper 1 analysis completed. Outputs: {args.output}")


if __name__ == "__main__":
    main()
