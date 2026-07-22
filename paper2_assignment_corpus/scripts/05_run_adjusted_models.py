#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2_assignments.adjusted_models import run_adjusted_models
from paper2_assignments.io_utils import read_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust OLS and fractional-logit sensitivity models for Paper 2.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = run_adjusted_models(read_table(args.input))
    with pd.ExcelWriter(output / "paper2_adjusted_models.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            if name == "analytic_data":
                continue
            table.to_excel(writer, sheet_name=name[:31], index=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"Adjusted-model outputs saved to {output.resolve()}")


if __name__ == "__main__":
    main()
