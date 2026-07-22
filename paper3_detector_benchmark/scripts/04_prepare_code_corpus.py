#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper3_benchmark.code_corpus import prepare_code_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone public GitHub repositories and create detector-ready code samples.")
    parser.add_argument("--urls", required=True, help="CSV containing a 'url' column.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_code_corpus(args.urls, args.output)
    print(f"Code-corpus outputs written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
