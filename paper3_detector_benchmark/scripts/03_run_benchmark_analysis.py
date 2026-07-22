#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper3_benchmark.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Paper 3 benchmark metrics and statistical tables.")
    parser.add_argument("--results", required=True, help="CSV or Excel detector-output table.")
    parser.add_argument("--output", required=True, help="Directory for analysis outputs.")
    parser.add_argument("--manifest", default=None, help="Optional file manifest for structural features.")
    parser.add_argument("--files-root", default=None, help="Root used to resolve relative manifest paths.")
    parser.add_argument("--sheet", default=0, help="Excel sheet name/index for the results table.")
    args = parser.parse_args()
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    tables = run_analysis(
        args.results,
        args.output,
        manifest_path=args.manifest,
        files_root=args.files_root,
        sheet_name=sheet,
    )
    print(f"Generated {len(tables)} analysis tables in {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
