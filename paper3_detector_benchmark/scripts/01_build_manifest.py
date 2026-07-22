#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper3_benchmark.io_utils import build_manifest_from_config, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark manifest from configured folders.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON configuration.")
    parser.add_argument("--output", required=True, help="Output manifest CSV path.")
    args = parser.parse_args()
    manifest = build_manifest_from_config(args.config)
    write_csv(manifest, args.output)
    print(f"Wrote {len(manifest)} manifest rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
