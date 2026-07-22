#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper2_assignments.io_utils import read_table
from paper2_assignments.pipeline import run_primary_analysis
from paper2_assignments.weights import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 2 primary descriptive and inferential analysis.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", help="Optional JSON detector-weight file.")
    args = parser.parse_args()
    run_primary_analysis(read_table(args.input), args.output, weights=load_weights(args.weights))
    print(f"Paper 2 primary outputs saved to {args.output}")


if __name__ == "__main__":
    main()
