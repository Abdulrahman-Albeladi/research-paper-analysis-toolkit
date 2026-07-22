from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[%()]", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


COLUMN_ALIASES = {
    "language": "language",
    "university": "university",
    "major": "major",
    "assignment_type": "assignment_type",
    "assignmenttype": "assignment_type",
    "year": "year_original",
    "year_group": "year_original",
    "year_grouping": "year_original",
    "file_name": "file_name",
    "filename": "file_name",
    "file_path": "file_path",
    "path": "file_path",
    "word_count": "word_count",
    "words": "word_count",
    "weighted_ai_rate": "combined_ai_rate_percent",
    "combined_ai_rate": "combined_ai_rate_percent",
    "combined_ai_rate_percent": "combined_ai_rate_percent",
    "combined_ai_score": "combined_ai_rate_percent",
    "combined_score": "combined_ai_rate_percent",
    "weighted_result": "combined_result",
    "gptzero_ai_rate_percent": "gptzero_ai_rate_percent",
    "gptzero_score_percent": "gptzero_ai_rate_percent",
    "pangram_ai_rate_percent": "pangram_ai_rate_percent",
    "pangram_score_percent": "pangram_ai_rate_percent",
    "sapling_ai_rate_percent": "sapling_ai_rate_percent",
    "sapling_score_percent": "sapling_ai_rate_percent",
    "isgen_ai_rate_percent": "isgen_ai_rate_percent",
    "isgen_score_percent": "isgen_ai_rate_percent",
    "gptzero_result": "gptzero_result",
    "pangram_result": "pangram_result",
    "sapling_result": "sapling_result",
    "isgen_result": "isgen_result",
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def standardize_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={column: normalize_column_name(column) for column in frame.columns}).copy()
    renamed = renamed.rename(columns={column: COLUMN_ALIASES[column] for column in renamed.columns if column in COLUMN_ALIASES})
    for column in ["language", "university", "major", "assignment_type", "year_original", "file_name", "file_path"]:
        if column in renamed.columns:
            renamed[column] = renamed[column].astype("string").str.strip()
            renamed.loc[renamed[column].isin(["", "nan", "None", "<NA>"]), column] = pd.NA
    for column in ["word_count", "combined_ai_rate_percent", "gptzero_ai_rate_percent", "pangram_ai_rate_percent", "sapling_ai_rate_percent", "isgen_ai_rate_percent"]:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed


def read_text(path: str | Path) -> str:
    path = Path(path)
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
