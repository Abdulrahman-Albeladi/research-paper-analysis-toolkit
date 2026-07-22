#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper2_assignments.collection import collect_detector_scores
from paper2_assignments.io_utils import read_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect commercial detector outputs for the Paper 2 corpus.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--files-root")
    parser.add_argument("--max-text-chars", type=int, default=50000)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = collect_detector_scores(
        read_table(args.manifest),
        args.output,
        files_root=args.files_root,
        max_text_chars=args.max_text_chars,
        sleep_between_calls=args.sleep,
        resume=not args.no_resume,
    )
    print(result["collection_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
