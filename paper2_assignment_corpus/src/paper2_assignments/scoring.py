from __future__ import annotations

import numpy as np
import pandas as pd

from .weights import DEFAULT_WEIGHTS

TOOLS = ("gptzero", "pangram", "sapling", "isgen")
SCORE_COLUMNS = [f"{tool}_ai_rate_percent" for tool in TOOLS]
RESULT_COLUMNS = [f"{tool}_result" for tool in TOOLS]


def canonical_language(value: object) -> str:
    text = str(value).strip().lower()
    if "arab" in text or "عرب" in text:
        return "Arabic"
    if "code" in text or "coding" in text or "program" in text:
        return "Code"
    return "English"


def normalize_label(value: object, score: float | None = None) -> str | float:
    if value is not None and not pd.isna(value):
        text = str(value).strip().lower()
        if "error" in text:
            return "ERROR"
        if "mixed" in text:
            return "MIXED"
        if "human" in text and "ai" not in text:
            return "HUMAN"
        if "ai" in text:
            return "AI"
    if score is not None and not pd.isna(score):
        return "AI" if float(score) >= 50 else "HUMAN"
    return np.nan


def combined_ai_rate(row: pd.Series, weights: dict[str, dict[str, float]] = DEFAULT_WEIGHTS) -> float:
    language = canonical_language(row.get("language", "English"))
    tool_weights = weights.get(language, weights["English"])
    numerator = 0.0
    denominator = 0.0
    for tool, weight in tool_weights.items():
        value = pd.to_numeric(pd.Series([row.get(f"{tool}_ai_rate_percent")]), errors="coerce").iloc[0]
        if pd.notna(value):
            numerator += float(value) * float(weight)
            denominator += float(weight)
    return numerator / denominator if denominator > 0 else np.nan


def add_combined_scores(frame: pd.DataFrame, weights: dict[str, dict[str, float]] = DEFAULT_WEIGHTS) -> pd.DataFrame:
    out = frame.copy()
    out["language"] = out["language"].map(canonical_language)
    for tool in TOOLS:
        score_column = f"{tool}_ai_rate_percent"
        result_column = f"{tool}_result"
        if score_column in out.columns:
            out[score_column] = pd.to_numeric(out[score_column], errors="coerce").clip(0, 100)
            out[f"{tool}_label_norm"] = [normalize_label(value, score) for value, score in zip(out.get(result_column, pd.Series(index=out.index, dtype=object)), out[score_column])]
    out["combined_ai_rate_percent"] = out.apply(combined_ai_rate, axis=1, weights=weights)
    out["combined_result"] = np.where(out["combined_ai_rate_percent"].ge(50), "AI", "HUMAN")
    return out
