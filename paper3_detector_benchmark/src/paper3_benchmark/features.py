from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .conflict import cliffs_delta
from .io_utils import read_text

AR_WORD_RE = re.compile(r"[\u0600-\u06FF]+")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str, language: str) -> list[str]:
    if language == "Arabic":
        return AR_WORD_RE.findall(text)
    if language == "English":
        return EN_WORD_RE.findall(text)
    return CODE_TOKEN_RE.findall(text)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def _heading_like(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if len(text) < 120 and re.match(r"^\d+(?:\.\d+)*[\).]?\s+\S", text):
        return True
    if re.match(r"^(chapter|section|abstract|introduction|method|methodology|results?|discussion|conclusion|references)\b", text, re.I):
        return True
    if re.match(r"^[A-Z][A-Z \-]{4,}$", text):
        return True
    return bool(re.match(r"^[\u0600-\u06FF\s]{2,}[:：]?$", text))


def _bullet_like(line: str) -> bool:
    text = line.strip()
    return bool(
        re.match(r"^([-*•])\s+", text)
        or re.match(r"^\d+[\.)]\s+", text)
        or re.match(r"^[A-Za-z][\.)]\s+", text)
    )


def extract_features(text: str, language: str) -> dict[str, float | int]:
    lines = text.splitlines()
    nonblank = [line for line in lines if line.strip()]
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
    tokens = _tokens(text, language)
    lower_tokens = [token.lower() for token in tokens]
    unique = len(set(lower_tokens))
    paragraph_lengths = [len(_tokens(paragraph, language)) for paragraph in paragraphs]
    repeated = Counter(line.strip() for line in nonblank if len(line.strip()) >= 5)
    repeated_line_count = sum(count - 1 for count in repeated.values() if count > 1)
    length = max(len(text), 1)
    nonblank_count = max(len(nonblank), 1)

    features: dict[str, float | int] = {
        "char_count": len(text),
        "line_count": len(lines),
        "nonblank_line_count": len(nonblank),
        "blank_line_count": len(lines) - len(nonblank),
        "paragraph_count": len(paragraphs),
        "word_count_extracted": len(tokens),
        "unique_word_count": unique,
        "type_token_ratio": unique / len(tokens) if tokens else np.nan,
        "avg_word_length": float(np.mean([len(token) for token in tokens])) if tokens else np.nan,
        "avg_line_length": float(np.mean([len(line) for line in nonblank])) if nonblank else np.nan,
        "avg_paragraph_words": float(np.mean(paragraph_lengths)) if paragraph_lengths else np.nan,
        "digit_density": len(re.findall(r"\d", text)) / length,
        "punctuation_density": len(re.findall(r"[^\w\s]", text, flags=re.UNICODE)) / length,
        "heading_density": sum(_heading_like(line) for line in nonblank) / nonblank_count,
        "bullet_density": sum(_bullet_like(line) for line in nonblank) / nonblank_count,
        "repeated_line_density": repeated_line_count / nonblank_count,
        "char_entropy": _entropy(text),
        "arabic_char_ratio": len(re.findall(r"[\u0600-\u06FF]", text)) / length,
        "latin_char_ratio": len(re.findall(r"[A-Za-z]", text)) / length,
    }
    if language == "Coding":
        features.update(
            {
                "code_import_count": len(re.findall(r"^\s*(?:import\s+|from\s+.+\s+import\s+)", text, re.M)),
                "code_comment_line_count": len(re.findall(r"^\s*(?:#|//|/\*|\*)", text, re.M)),
                "code_function_like_count": len(re.findall(r"\b(?:def|function|func|void|int|float|double|public|private|protected|static)\b", text)),
                "code_class_like_count": len(re.findall(r"\bclass\b", text)),
                "code_brace_count": text.count("{") + text.count("}"),
                "code_semicolon_count": text.count(";"),
                "code_indent_line_count": sum(bool(re.match(r"^\s{2,}\S", line)) for line in nonblank),
                "code_blank_line_ratio": (len(lines) - len(nonblank)) / max(len(lines), 1),
            }
        )
    return features


def build_feature_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        path = row["file_path"]
        text = read_text(path)
        rows.append(
            {
                "Language": row["language"],
                "Label": row["label"],
                "file_name": row["file_name"],
                "file_path": path,
                **extract_features(text, row["language"]),
            }
        )
    return pd.DataFrame(rows)


def compare_structural_features(merged: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        column
        for column in merged.columns
        if column
        in {
            "word_count", "word_count_extracted", "char_count", "line_count", "nonblank_line_count",
            "blank_line_count", "paragraph_count", "unique_word_count", "type_token_ratio",
            "avg_word_length", "avg_line_length", "avg_paragraph_words", "digit_density",
            "punctuation_density", "heading_density", "bullet_density", "repeated_line_density",
            "char_entropy", "text_score_mean", "text_score_std", "text_score_min", "text_score_max",
            "text_score_range", "text_score_margin_from_50_abs", "arabic_char_ratio", "latin_char_ratio",
            "code_import_count", "code_comment_line_count", "code_function_like_count",
            "code_class_like_count", "code_brace_count", "code_semicolon_count",
            "code_indent_line_count", "code_blank_line_ratio",
        }
    ]
    specifications = [
        ("Arabic", "AI-Free", "is_text_conflict"),
        ("Arabic", "AI-Generated", "is_text_conflict"),
        ("Arabic", "Humanized AI", "is_text_conflict"),
        ("English", "AI-Free", "is_text_conflict"),
        ("English", "Humanized AI", "is_text_conflict"),
        ("Coding", "AI-Generated_or_Humanized", "coding_pangram_correct"),
    ]
    all_rows: list[dict[str, object]] = []
    for language, label, group_col in specifications:
        if language == "Coding":
            subset = merged[(merged["Language"] == language) & merged["Label"].isin(["AI-Generated", "Humanized AI"])]
        else:
            subset = merged[(merged["Language"] == language) & (merged["Label"] == label)]
        for feature in numeric_features:
            group1 = pd.to_numeric(subset.loc[subset[group_col] == True, feature], errors="coerce").dropna()
            group0 = pd.to_numeric(subset.loc[subset[group_col] == False, feature], errors="coerce").dropna()
            p_value = np.nan
            if len(group1) >= 2 and len(group0) >= 2:
                p_value = float(mannwhitneyu(group1, group0, alternative="two-sided").pvalue)
            all_rows.append(
                {
                    "Comparison": f"{language}_{label}",
                    "Feature": feature,
                    "Group1_N": len(group1),
                    "Group0_N": len(group0),
                    "Group1_Mean": group1.mean() if len(group1) else np.nan,
                    "Group0_Mean": group0.mean() if len(group0) else np.nan,
                    "Group1_Median": group1.median() if len(group1) else np.nan,
                    "Group0_Median": group0.median() if len(group0) else np.nan,
                    "Cliffs_Delta": cliffs_delta(group1, group0),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(all_rows)
    result["BH_Adjusted_p_value"] = np.nan
    for comparison, index in result.groupby("Comparison").groups.items():
        valid_index = [i for i in index if pd.notna(result.at[i, "p_value"])]
        if valid_index:
            result.loc[valid_index, "BH_Adjusted_p_value"] = multipletests(
                result.loc[valid_index, "p_value"], method="fdr_bh"
            )[1]
    return result.sort_values(["Comparison", "BH_Adjusted_p_value", "p_value"], na_position="last")
