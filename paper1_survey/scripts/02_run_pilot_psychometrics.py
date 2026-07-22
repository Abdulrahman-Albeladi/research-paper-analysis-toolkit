#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper1_survey.io import read_table
from paper1_survey.psychometrics import pilot_scale_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare psychometrics of the old and edited pilot surveys.")
    parser.add_argument("--old-survey", required=True)
    parser.add_argument("--edited-survey", required=True)
    parser.add_argument("--old-scales", required=True, help="JSON mapping scale names to old-survey item columns.")
    parser.add_argument("--edited-scales", required=True, help="JSON mapping scale names to edited-survey item columns.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    old_scales = json.loads(Path(args.old_scales).read_text(encoding="utf-8"))
    edited_scales = json.loads(Path(args.edited_scales).read_text(encoding="utf-8"))
    old_summary, old_loadings = pilot_scale_diagnostics(read_table(args.old_survey), old_scales)
    edited_summary, edited_loadings = pilot_scale_diagnostics(read_table(args.edited_survey), edited_scales)
    old_summary.insert(0, "survey_version", "old")
    edited_summary.insert(0, "survey_version", "edited")
    summary = pd.concat([old_summary, edited_summary], ignore_index=True)
    old_loadings.insert(0, "survey_version", "old")
    edited_loadings.insert(0, "survey_version", "edited")
    loadings = pd.concat([old_loadings, edited_loadings], ignore_index=True)
    summary.to_csv(output / "pilot_psychometric_summary.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(output / "pilot_factor_loadings.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output / "pilot_psychometric_comparison.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        loadings.to_excel(writer, sheet_name="Loadings", index=False)
    print(f"Saved pilot outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
