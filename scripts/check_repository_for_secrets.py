#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "Hard-coded detector key": re.compile(
        r"(?i)\b(?:GPTZERO_API_KEY|PANGRAM_API_KEY|SAPLING_API_KEY|ISGEN_RAPIDAPI_KEY|OPENAI_API_KEY)\s*=\s*['\"][^'\"]{8,}['\"]"
    ),
    "Hard-coded password": re.compile(r"(?i)\b(?:password|passwd|pw)\s*=\s*['\"][^'\"]{8,}['\"]"),
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".csv", ".cff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if likely credentials are present in repository text files.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    findings: list[tuple[Path, str, int]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((path, name, line))
    if findings:
        for path, name, line in findings:
            print(f"{path}:{line}: {name}")
        raise SystemExit(1)
    print("No likely embedded credentials found.")


if __name__ == "__main__":
    main()
