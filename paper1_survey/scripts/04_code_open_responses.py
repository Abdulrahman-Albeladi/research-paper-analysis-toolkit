#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper1_survey.io import read_table
from paper1_survey.qualitative import code_open_responses
from paper1_survey.schema import DEFAULT_SCHEMA, SurveySchema


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised multi-label coding of de-identified Paper 1 open responses.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--model")
    args = parser.parse_args()
    schema = SurveySchema.from_json(args.schema) if args.schema else DEFAULT_SCHEMA
    code_open_responses(read_table(args.input), args.output, schema, model=args.model)
    print(f"Open-response outputs saved to {args.output}")


if __name__ == "__main__":
    main()
