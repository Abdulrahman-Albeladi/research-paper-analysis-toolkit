#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper2_assignments.io_utils import write_csv
from paper2_assignments.preprocessing import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Paper 2 assignment manifest from a folder tree.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    write_csv(manifest, args.output)
    print(f"Manifest rows: {len(manifest)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
