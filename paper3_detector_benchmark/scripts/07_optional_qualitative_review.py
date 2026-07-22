#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper3_benchmark.qualitative import run_qualitative_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional OpenAI-assisted qualitative review of selected conflict cases.")
    parser.add_argument("--analysis-master", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-group", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    table = pd.read_csv(args.analysis_master, encoding="utf-8-sig")
    result = run_qualitative_review(
        table,
        args.output,
        max_files_per_group=args.max_per_group,
        model=args.model,
    )
    print(f"Wrote {len(result)} qualitative-review rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
