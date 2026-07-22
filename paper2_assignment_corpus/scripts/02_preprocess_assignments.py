#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper2_assignments.io_utils import read_table, write_csv
from paper2_assignments.preprocessing import preprocess_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and deterministically clean Paper 2 assignment text.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--files-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--minimum-words", type=int, default=300)
    parser.add_argument("--keep-non-authorial", action="store_true", help="Do not remove obvious reference/TOC/page-number blocks.")
    args = parser.parse_args()
    result = preprocess_manifest(
        read_table(args.manifest),
        files_root=args.files_root,
        output_root=args.output_root,
        minimum_words=args.minimum_words,
        remove_non_authorial=not args.keep_non_authorial,
    )
    write_csv(result, args.output_manifest)
    print(result["preprocess_status"].value_counts(dropna=False).to_string())
    print(f"Saved: {args.output_manifest}")


if __name__ == "__main__":
    main()
