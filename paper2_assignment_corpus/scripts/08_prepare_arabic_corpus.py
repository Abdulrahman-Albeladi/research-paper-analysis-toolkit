#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper2_assignments.arabic import arabic_character_ratio, normalize_arabic_text, repair_split_arabic_words_with_openai, suspicious_token_ratio
from paper2_assignments.io_utils import read_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Arabic TXT files and report likely corruption.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--openai-repair", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    rows = []
    for source in sorted(input_root.rglob("*.txt")):
        relative = source.relative_to(input_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = normalize_arabic_text(read_text(source))
        if args.openai_repair:
            text = repair_split_arabic_words_with_openai(text, model=args.model)
        destination.write_text(text, encoding="utf-8")
        rows.append({"relative_path": relative.as_posix(), "arabic_character_ratio": arabic_character_ratio(text), "suspicious_token_ratio": suspicious_token_ratio(text), "word_count": len(text.split())})
    report = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_root / "arabic_quality_report.csv", index=False, encoding="utf-8-sig")
    print(f"Processed {len(report)} Arabic files.")


if __name__ == "__main__":
    main()
