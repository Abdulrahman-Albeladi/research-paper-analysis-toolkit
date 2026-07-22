#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper3_benchmark.humanizer import process_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Humanize English generated text through the external service used in the study.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_directory(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
