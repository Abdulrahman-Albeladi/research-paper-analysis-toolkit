#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2_assignments.code_corpus import prepare_github_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create de-identified multi-file code submissions from public GitHub repository URLs.")
    parser.add_argument("--url-csv", required=True, help="CSV containing file_id and repository_url columns.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=6000)
    args = parser.parse_args()
    frame = pd.read_csv(args.url_csv)
    required = {"file_id", "repository_url"}
    if not required.issubset(frame.columns):
        raise ValueError(f"URL CSV must contain {sorted(required)}")
    output_root = Path(args.output_root)
    rows = []
    for item in frame.to_dict(orient="records"):
        destination = output_root / f"{item['file_id']}.txt"
        try:
            prepare_github_submission(item["repository_url"], destination, args.max_files, args.max_words)
            rows.append({"file_id": item["file_id"], "repository_url": item["repository_url"], "output_path": destination.as_posix(), "status": "complete", "error": ""})
        except Exception as error:
            rows.append({"file_id": item["file_id"], "repository_url": item["repository_url"], "output_path": "", "status": "error", "error": repr(error)})
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "code_preparation_report.csv", index=False, encoding="utf-8-sig")
    print(f"Processed {len(rows)} repository records.")


if __name__ == "__main__":
    main()
