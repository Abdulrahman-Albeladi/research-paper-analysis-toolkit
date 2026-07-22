from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .schema import TEXT_TOOLS


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    differences = np.subtract.outer(a, b)
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def _pattern(row: pd.Series) -> pd.Series:
    if row["Language"] not in {"Arabic", "English"}:
        return pd.Series(
            {
                "pattern_type": "",
                "lone_dissenter_tool": "",
                "pair_ai": "",
                "pair_human": "",
                "unanimous_label": "",
                "conflict_signature": "",
            }
        )
    votes = {tool: row.get(f"{tool}_pred_label") for tool in TEXT_TOOLS}
    votes = {tool: value for tool, value in votes.items() if value in {"AI", "Human"}}
    if len(votes) != 4:
        return pd.Series(
            {
                "pattern_type": "Incomplete",
                "lone_dissenter_tool": "",
                "pair_ai": "",
                "pair_human": "",
                "unanimous_label": "",
                "conflict_signature": "Incomplete detector results",
            }
        )
    counts = Counter(votes.values())
    if len(counts) == 1:
        label = next(iter(counts))
        return pd.Series(
            {
                "pattern_type": "Unanimous",
                "lone_dissenter_tool": "",
                "pair_ai": "",
                "pair_human": "",
                "unanimous_label": label,
                "conflict_signature": f"Unanimous {label}",
            }
        )
    if sorted(counts.values()) == [1, 3]:
        minority = min(counts, key=counts.get)
        tool = next(name for name, label in votes.items() if label == minority)
        return pd.Series(
            {
                "pattern_type": "Lone dissenter",
                "lone_dissenter_tool": tool,
                "pair_ai": "",
                "pair_human": "",
                "unanimous_label": "",
                "conflict_signature": f"{tool} lone dissenter",
            }
        )
    ai_pair = [tool for tool, label in votes.items() if label == "AI"]
    human_pair = [tool for tool, label in votes.items() if label == "Human"]
    if len(ai_pair) == 2 and len(human_pair) == 2:
        ai_text = " + ".join(ai_pair)
        human_text = " + ".join(human_pair)
        return pd.Series(
            {
                "pattern_type": "2 vs 2 split",
                "lone_dissenter_tool": "",
                "pair_ai": ai_text,
                "pair_human": human_text,
                "unanimous_label": "",
                "conflict_signature": f"{ai_text} vs {human_text}",
            }
        )
    return pd.Series(
        {
            "pattern_type": "Other",
            "lone_dissenter_tool": "",
            "pair_ai": "",
            "pair_human": "",
            "unanimous_label": "",
            "conflict_signature": "Other",
        }
    )


def add_conflict_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_mask = out["Language"].isin(["Arabic", "English"])

    def votes(row: pd.Series) -> list[str]:
        if row["Language"] not in {"Arabic", "English"}:
            return []
        return [row[f"{tool}_pred_label"] for tool in TEXT_TOOLS if row.get(f"{tool}_pred_label") in {"AI", "Human"}]

    out["text_votes"] = out.apply(votes, axis=1)
    out["is_text_conflict"] = out["text_votes"].map(lambda values: len(values) >= 2 and len(set(values)) > 1)
    out["is_text_unanimous"] = out["text_votes"].map(lambda values: len(values) == 4 and len(set(values)) == 1)
    out["n_votes_ai"] = out["text_votes"].map(lambda values: values.count("AI"))
    out["n_votes_human"] = out["text_votes"].map(lambda values: values.count("Human"))

    score_columns = [f"{tool}_score_percent" for tool in TEXT_TOOLS]
    score_values = out[score_columns].apply(pd.to_numeric, errors="coerce")
    out["text_score_mean"] = np.where(text_mask, score_values.mean(axis=1, skipna=True), np.nan)
    out["text_score_std"] = np.where(text_mask, score_values.std(axis=1, skipna=True, ddof=0), np.nan)
    out["text_score_min"] = np.where(text_mask, score_values.min(axis=1, skipna=True), np.nan)
    out["text_score_max"] = np.where(text_mask, score_values.max(axis=1, skipna=True), np.nan)
    out["text_score_range"] = np.where(text_mask, out["text_score_max"] - out["text_score_min"], np.nan)
    out["text_score_margin_from_50_abs"] = np.where(text_mask, (out["text_score_mean"] - 50.0).abs(), np.nan)
    out["coding_pangram_correct"] = np.where(
        out["Language"].eq("Coding"), out["pangram_pred_label"] == out["truth_class"], np.nan
    )

    pattern_fields = out.apply(_pattern, axis=1)
    for column in pattern_fields.columns:
        out[column] = pattern_fields[column]
    return out


def conflict_counts(df: pd.DataFrame) -> pd.DataFrame:
    text = df[df["Language"].isin(["Arabic", "English"])].copy()
    return (
        text.groupby(["Language", "Label"], observed=True)
        .agg(
            N=("file_name", "size"),
            Conflicting=("is_text_conflict", "sum"),
            Mean_Score_Range=("text_score_range", "mean"),
            Median_Score_Range=("text_score_range", "median"),
        )
        .reset_index()
        .assign(Conflict_Percent=lambda table: 100 * table["Conflicting"] / table["N"])
    )


def pattern_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["Language"].isin(["Arabic", "English"])]
        .groupby(["Language", "Label", "pattern_type", "conflict_signature"], observed=True)
        .size()
        .reset_index(name="N")
        .sort_values(["Language", "Label", "N"], ascending=[True, True, False])
    )


def dominant_conflict_signatures(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    counts = pattern_counts(df)
    counts = counts[counts["pattern_type"].isin(["Lone dissenter", "2 vs 2 split"])]
    return counts.groupby(["Language", "Label"], observed=True).head(top_n).reset_index(drop=True)


def uncertainty_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for language in ("Arabic", "English"):
        for label in ("AI-Free", "AI-Generated", "Humanized AI"):
            subset = df[(df["Language"] == language) & (df["Label"] == label)]
            conflict = pd.to_numeric(
                subset.loc[subset["is_text_conflict"], "text_score_range"], errors="coerce"
            ).dropna()
            nonconflict = pd.to_numeric(
                subset.loc[~subset["is_text_conflict"], "text_score_range"], errors="coerce"
            ).dropna()
            p_value = np.nan
            if len(conflict) >= 2 and len(nonconflict) >= 2:
                p_value = float(mannwhitneyu(conflict, nonconflict, alternative="two-sided").pvalue)
            rows.append(
                {
                    "Language": language,
                    "Label": label,
                    "Conflict_N": len(conflict),
                    "NonConflict_N": len(nonconflict),
                    "Conflict_Mean_Range": conflict.mean() if len(conflict) else np.nan,
                    "NonConflict_Mean_Range": nonconflict.mean() if len(nonconflict) else np.nan,
                    "Conflict_Median_Range": conflict.median() if len(conflict) else np.nan,
                    "NonConflict_Median_Range": nonconflict.median() if len(nonconflict) else np.nan,
                    "Cliffs_Delta": cliffs_delta(conflict, nonconflict),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result["BH_Adjusted_p_value"] = np.nan
    if valid.any():
        result.loc[valid, "BH_Adjusted_p_value"] = multipletests(
            result.loc[valid, "p_value"], method="fdr_bh"
        )[1]
    return result


def coding_accuracy_by_label(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["Language"] == "Coding"]
    return (
        subset.groupby("Label", observed=True)["coding_pangram_correct"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"size": "N", "sum": "Correct", "mean": "Accuracy"})
        .assign(Accuracy_Percent=lambda table: 100 * table["Accuracy"])
        .drop(columns="Accuracy")
    )
