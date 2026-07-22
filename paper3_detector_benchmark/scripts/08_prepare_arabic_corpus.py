#!/usr/bin/env python3
from __future__ import annotations

import argparse

from paper3_benchmark.arabic_preprocessing import process_pdf_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract, OCR if needed, and deletion-only trim Arabic PDF text.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ocr", action="store_true", help="Use OpenAI image OCR instead of embedded PDF text.")
    parser.add_argument("--no-openai-trim", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_pdf_directory(
        args.input,
        args.output,
        use_ocr=args.ocr,
        use_openai_trim=not args.no_openai_trim,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
