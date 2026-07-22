from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

TEXT_TOOLS: tuple[str, ...] = ("gptzero", "pangram", "sapling", "isgen")
CODE_TOOLS: tuple[str, ...] = ("pangram",)
ALL_TOOLS: tuple[str, ...] = TEXT_TOOLS
LANGUAGES: tuple[str, ...] = ("Arabic", "English", "Coding")
LABELS: tuple[str, ...] = ("AI-Free", "AI-Generated", "Humanized AI")


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_language(value: object) -> str:
    text = _clean(value)
    mapping = {
        "arabic": "Arabic",
        "arabic text": "Arabic",
        "english": "English",
        "english text": "English",
        "code": "Coding",
        "coding": "Coding",
        "source code": "Coding",
        "programming": "Coding",
    }
    return mapping.get(text, str(value).strip() if value is not None else "")


def canonical_label(value: object) -> str:
    text = _clean(value)
    if text in {"ai free", "human", "human written", "fully human", "presumed human"}:
        return "AI-Free"
    if text in {"ai generated", "ai full", "full ai", "fully ai", "pure ai", "generated ai"}:
        return "AI-Generated"
    if text in {"humanized ai", "humanized", "ai humanized", "post generation humanized"}:
        return "Humanized AI"
    return str(value).strip() if value is not None else ""


def true_class(label: object) -> str | None:
    canonical = canonical_label(label)
    if canonical == "AI-Free":
        return "Human"
    if canonical in {"AI-Generated", "Humanized AI"}:
        return "AI"
    return None


def tools_for_language(language: object) -> tuple[str, ...]:
    return CODE_TOOLS if canonical_language(language) == "Coding" else TEXT_TOOLS


def normalize_header(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def first_matching_column(
    columns: Iterable[object], aliases: Iterable[str], *, required: bool = False
) -> str | None:
    mapping = {normalize_header(column): str(column) for column in columns}
    for alias in aliases:
        match = mapping.get(normalize_header(alias))
        if match is not None:
            return match
    if required:
        raise KeyError(f"None of the required columns were found: {list(aliases)}")
    return None


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
