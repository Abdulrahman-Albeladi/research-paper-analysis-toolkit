from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cliffs_delta(x, y) -> float:
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    b = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = sum(np.sum(value > b) for value in a)
    less = sum(np.sum(value < b) for value in a)
    return float((greater - less) / (len(a) * len(b)))


def group_summary(frame: pd.DataFrame, group_column: str, value_column: str = "combined_ai_rate_percent") -> pd.DataFrame:
    return (
        frame.groupby(group_column, dropna=False)
        .agg(
            n=(value_column, "count"),
            mean=(value_column, "mean"),
            median=(value_column, "median"),
            sd=(value_column, "std"),
            mean_word_count=("word_count", "mean") if "word_count" in frame.columns else (value_column, "size"),
        )
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def kruskal_wallis(frame: pd.DataFrame, group_column: str, value_column: str = "combined_ai_rate_percent", min_n: int = 3) -> dict[str, object]:
    groups = []
    labels = []
    for label, subset in frame.groupby(group_column, dropna=True):
        values = pd.to_numeric(subset[value_column], errors="coerce").dropna()
        if len(values) >= min_n:
            labels.append(label)
            groups.append(values.to_numpy())
    if len(groups) < 2:
        return {"grouping_variable": group_column, "groups_tested": len(groups), "H": np.nan, "p": np.nan, "status": "insufficient groups"}
    statistic, p_value = stats.kruskal(*groups)
    return {"grouping_variable": group_column, "groups_tested": len(groups), "H": float(statistic), "p": float(p_value), "status": "ok"}


def pairwise_mann_whitney(frame: pd.DataFrame, group_column: str, value_column: str = "combined_ai_rate_percent", min_n: int = 3) -> pd.DataFrame:
    grouped = {
        label: pd.to_numeric(subset[value_column], errors="coerce").dropna().to_numpy()
        for label, subset in frame.groupby(group_column, dropna=True)
    }
    rows: list[dict[str, object]] = []
    for first, second in combinations(sorted(grouped, key=str), 2):
        x, y = grouped[first], grouped[second]
        if len(x) < min_n or len(y) < min_n:
            continue
        statistic, p_value = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        rows.append(
            {
                "grouping_variable": group_column,
                "group_1": first,
                "group_2": second,
                "n_1": len(x),
                "n_2": len(y),
                "mean_1": float(np.mean(x)),
                "mean_2": float(np.mean(y)),
                "mean_difference": float(np.mean(x) - np.mean(y)),
                "U": float(statistic),
                "p_raw": float(p_value),
                "cliffs_delta": cliffs_delta(x, y),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = multipletests(out["p_raw"], method="fdr_bh")[1]
        out["significant_fdr"] = out["p_fdr"] < 0.05
    return out


def word_count_correlations(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in value_columns:
        if column not in frame.columns:
            continue
        data = frame[["word_count", column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 3:
            continue
        spearman = stats.spearmanr(data["word_count"], data[column])
        pearson = stats.pearsonr(data["word_count"], data[column])
        rows.append(
            {
                "score": column,
                "n": len(data),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
            }
        )
    return pd.DataFrame(rows)


def detector_language_comparisons(frame: pd.DataFrame, detectors: tuple[str, ...] = ("gptzero", "pangram", "sapling", "isgen")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tool in detectors:
        column = f"{tool}_ai_rate_percent"
        if column not in frame.columns:
            continue
        english = pd.to_numeric(frame.loc[frame["language"] == "English", column], errors="coerce").dropna()
        arabic = pd.to_numeric(frame.loc[frame["language"] == "Arabic", column], errors="coerce").dropna()
        if len(english) < 3 or len(arabic) < 3:
            continue
        statistic, p_value = stats.mannwhitneyu(english, arabic, alternative="two-sided", method="auto")
        rows.append(
            {
                "detector": tool,
                "english_n": len(english),
                "arabic_n": len(arabic),
                "english_mean": float(english.mean()),
                "arabic_mean": float(arabic.mean()),
                "english_median": float(english.median()),
                "arabic_median": float(arabic.median()),
                "mean_difference_english_minus_arabic": float(english.mean() - arabic.mean()),
                "U": float(statistic),
                "p_raw": float(p_value),
                "cliffs_delta": cliffs_delta(english, arabic),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = multipletests(out["p_raw"], method="fdr_bh")[1]
    return out
