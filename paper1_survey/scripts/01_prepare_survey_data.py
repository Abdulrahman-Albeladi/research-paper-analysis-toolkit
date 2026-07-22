#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper1_survey.cleaning import prepare_survey
from paper1_survey.io import read_table, write_table
from paper1_survey.schema import DEFAULT_SCHEMA, SurveySchema


def parse_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map and quality-check the Paper 1 survey dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--schema", help="Optional JSON file overriding survey column names.")
    parser.add_argument("--drop-flagged", action="store_true", help="Drop every row carrying any quality flag.")
    parser.add_argument("--drop-indices", default="", help="Comma-separated row indices explicitly approved for removal.")
    args = parser.parse_args()

    schema = SurveySchema.from_json(args.schema) if args.schema else DEFAULT_SCHEMA
    cleaned, quality = prepare_survey(
        read_table(args.input),
        schema,
        drop_flagged=args.drop_flagged,
        explicit_drop_indices=parse_indices(args.drop_indices),
    )
    write_table(cleaned, args.output)
    write_table(quality, args.quality_report)
    print(f"Saved mapped survey: {Path(args.output).resolve()}")
    print(f"Saved quality report: {Path(args.quality_report).resolve()}")
    print(f"Rows retained: {len(cleaned)}")


if __name__ == "__main__":
    main()
