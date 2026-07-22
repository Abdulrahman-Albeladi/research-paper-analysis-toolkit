from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from .io_utils import read_text

FEATURE_COLUMNS = [
    "template_artifacts",
    "formatting_artifacts",
    "reference_heavy_text",
    "formulaic_structure",
    "mixed_writing_style",
    "non_native_english_signals",
    "technical_jargon_density",
    "paraphrase_like_phrasing",
    "generic_or_overpolished_style",
    "abrupt_style_shifts",
    "length_or_sectioning_indicators",
]

FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": FEATURE_COLUMNS + ["confidence", "brief_notes"],
    "properties": {
        **{name: {"type": "boolean"} for name in FEATURE_COLUMNS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "brief_notes": {"type": "string", "maxLength": 500},
    },
}

IDENTIFIER_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)"),
    re.compile(r"\b(?:Taif|Taibah|KFUPM|Jeddah|King Fahd|University of Jeddah)\b", re.I),
]


def deidentify_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    for pattern in IDENTIFIER_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_excerpt(text: str, max_characters: int = 4500) -> str:
    text = deidentify_text(text)
    if len(text) <= max_characters:
        return text
    half = max_characters // 2
    return text[:half] + "\n\n[...middle omitted... ]\n\n" + text[-half:]


def match_comparison_group(frame: pd.DataFrame, flag_column: str = "high_impact_conflict") -> pd.DataFrame:
    high = frame[frame[flag_column]].copy()
    pool = frame[~frame[flag_column]].copy()
    selected: list[pd.Series] = []
    used: set[int] = set()
    for _, row in high.iterrows():
        candidates = pool[(pool["language"] == row["language"]) & (pool["assignment_type"] == row["assignment_type"]) & (~pool.index.isin(used))]
        if candidates.empty:
            candidates = pool[(pool["language"] == row["language"]) & (~pool.index.isin(used))]
        if candidates.empty:
            candidates = pool[~pool.index.isin(used)]
        if candidates.empty:
            break
        distance = (pd.to_numeric(candidates["word_count"], errors="coerce") - float(row["word_count"])).abs()
        chosen_index = distance.idxmin()
        used.add(int(chosen_index))
        selected.append(pool.loc[chosen_index])
    comparison = pd.DataFrame(selected)
    high = high.assign(comparison_group="high_impact_conflict")
    comparison = comparison.assign(comparison_group="matched_non_high_conflict")
    return pd.concat([high, comparison], ignore_index=True)


def code_excerpt(client, *, model: str, excerpt: str, context: dict[str, object], max_retries: int = 5) -> dict[str, object]:
    prompt = (
        "Code only observable textual and formatting features in this de-identified university-assignment excerpt. "
        "Do not infer or state whether it is AI-generated and do not make an authorship judgment.\n\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}\n\nExcerpt:\n{excerpt}"
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                text={"format": {"type": "json_schema", "name": "assignment_textual_features", "schema": FEATURE_SCHEMA, "strict": True}},
            )
            return json.loads(response.output_text)
        except Exception as error:  # pragma: no cover - network-dependent
            last_error = error
            time.sleep(min(8.0, 1.5 ** attempt))
    raise RuntimeError(f"Feature coding failed: {last_error}")


def run_feature_coding(frame: pd.DataFrame, output_dir: str | Path, *, model: str | None = None, max_files: int | None = None) -> pd.DataFrame:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install the API dependency group.") from error
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched = match_comparison_group(frame)
    matched = matched[matched["file_path"].notna()].copy()
    if max_files is not None:
        matched = matched.head(max_files)
    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    rows: list[dict[str, object]] = []
    for row in matched.to_dict(orient="records"):
        try:
            excerpt = build_excerpt(read_text(row["file_path"]))
            coded = code_excerpt(
                client,
                model=model,
                excerpt=excerpt,
                context={
                    "language": row.get("language"),
                    "assignment_type": row.get("assignment_type"),
                    "word_count": row.get("word_count"),
                    "detector_score_range": row.get("detector_score_range"),
                },
            )
            rows.append({"file_id": row.get("file_id"), "comparison_group": row["comparison_group"], **coded})
        except Exception as error:
            rows.append({"file_id": row.get("file_id"), "comparison_group": row["comparison_group"], "coding_error": repr(error)})
        pd.DataFrame(rows).to_csv(output_dir / "textual_feature_codes.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def compare_feature_prevalence(codes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in codes.columns:
            continue
        table = pd.crosstab(codes["comparison_group"], codes[feature].astype("boolean"))
        if table.shape != (2, 2):
            continue
        expected_small = (table.to_numpy() < 5).any()
        if expected_small:
            _, p_value = fisher_exact(table.to_numpy())
            test = "Fisher exact"
            statistic = np.nan
        else:
            statistic, p_value, _, _ = chi2_contingency(table.to_numpy())
            test = "Chi-square"
        high_percent = 100 * codes.loc[codes["comparison_group"] == "high_impact_conflict", feature].astype(bool).mean()
        comparison_percent = 100 * codes.loc[codes["comparison_group"] == "matched_non_high_conflict", feature].astype(bool).mean()
        rows.append({"feature": feature, "test": test, "statistic": statistic, "p_raw": float(p_value), "high_conflict_percent": high_percent, "matched_percent": comparison_percent})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = multipletests(out["p_raw"], method="fdr_bh")[1]
        out["significant_fdr"] = out["p_fdr"] < 0.05
    return out
