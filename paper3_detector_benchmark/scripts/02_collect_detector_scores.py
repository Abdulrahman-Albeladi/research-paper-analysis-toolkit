#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper3_benchmark.collection import collect_detector_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect commercial detector scores for the benchmark manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--files-root", default=None)
    parser.add_argument("--max-text-chars", type=int, default=50_000)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = collect_detector_scores(
        args.manifest,
        args.output,
        files_root=args.files_root,
        max_text_chars=args.max_text_chars,
        sleep_between_calls=args.sleep,
        resume=not args.no_resume,
    )
    print(f"Saved {len(result)} result rows to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
