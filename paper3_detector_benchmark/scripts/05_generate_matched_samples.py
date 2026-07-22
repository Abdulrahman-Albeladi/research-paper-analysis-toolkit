#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper3_benchmark.generation import process_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-prompt and generate matched Arabic or code samples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--modality", required=True, choices=["arabic", "code"])
    parser.add_argument("--humanized", default=None, help="Optional output for OpenAI-humanized code.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_directory(
        args.input,
        args.prompts,
        args.generated,
        modality=args.modality,
        humanized_dir=args.humanized,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
