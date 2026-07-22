from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .scoring import TOOLS, normalize_label


def add_conflict_fields(
    frame: pd.DataFrame,
    *,
    broad_range_threshold: float = 40.0,
    low_threshold: float = 30.0,
    high_threshold: float = 70.0,
    combined_low: float = 40.0,
    combined_high: float = 60.0,
) -> pd.DataFrame:
    out = frame.copy()
    score_columns = [f"{tool}_ai_rate_percent" for tool in TOOLS if f"{tool}_ai_rate_percent" in out.columns]
    for column in score_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    score_matrix = out[score_columns]
    out["n_detector_scores_available"] = score_matrix.notna().sum(axis=1)
    out["detector_score_min"] = score_matrix.min(axis=1, skipna=True)
    out["detector_score_max"] = score_matrix.max(axis=1, skipna=True)
    out["detector_score_range"] = out["detector_score_max"] - out["detector_score_min"]
    out["detector_score_variance"] = score_matrix.var(axis=1, skipna=True)

    label_columns: list[str] = []
    for tool in TOOLS:
        score_column = f"{tool}_ai_rate_percent"
        result_column = f"{tool}_result"
        if score_column not in out.columns:
            continue
        normalized_column = f"{tool}_label_norm"
        supplied = out[result_column] if result_column in out.columns else pd.Series(index=out.index, dtype=object)
        out[normalized_column] = [normalize_label(value, score) for value, score in zip(supplied, out[score_column])]
        label_columns.append(normalized_column)

    def labels_disagree(row: pd.Series) -> bool:
        labels = {value for value in row[label_columns].dropna() if value not in {"ERROR", "MIXED"}}
        return len(labels) > 1

    out["label_conflict"] = out.apply(labels_disagree, axis=1) if label_columns else False
    out["rate_conflict"] = out["detector_score_range"].ge(broad_range_threshold) & out["n_detector_scores_available"].ge(2)
    out["broad_conflict"] = out["label_conflict"] | out["rate_conflict"]
    out["high_impact_conflict"] = (
        out["detector_score_min"].le(low_threshold)
        & out["detector_score_max"].ge(high_threshold)
        & out["combined_ai_rate_percent"].between(combined_low, combined_high, inclusive="both")
        & out["n_detector_scores_available"].ge(2)
    )
    return out


def pairwise_detector_differences(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = [tool for tool in TOOLS if f"{tool}_ai_rate_percent" in frame.columns]
    for first, second in combinations(available, 2):
        columns = [f"{first}_ai_rate_percent", f"{second}_ai_rate_percent"]
        data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
        difference = (data[columns[0]] - data[columns[1]]).abs()
        rows.append(
            {
                "detector_pair": f"{first} vs {second}",
                "n": len(data),
                "mean_absolute_difference": float(difference.mean()),
                "median_absolute_difference": float(difference.median()),
                "sd_absolute_difference": float(difference.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def conflict_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "Files analyzed", "value": len(frame)},
        {"metric": "Label-conflict files", "value": int(frame["label_conflict"].sum())},
        {"metric": "Rate-conflict files", "value": int(frame["rate_conflict"].sum())},
        {"metric": "Files meeting either broad rule", "value": int(frame["broad_conflict"].sum())},
        {"metric": "High-impact conflict files", "value": int(frame["high_impact_conflict"].sum())},
        {"metric": "Mean detector-score range", "value": float(frame["detector_score_range"].mean())},
        {"metric": "Median detector-score range", "value": float(frame["detector_score_range"].median())},
        {"metric": "SD detector-score range", "value": float(frame["detector_score_range"].std(ddof=1))},
    ]
    for threshold in (20, 30, 40):
        rows.append({"metric": f"Files with detector-score range >= {threshold}", "value": int(frame["detector_score_range"].ge(threshold).sum())})
    return pd.DataFrame(rows)


def conflict_by_group(frame: pd.DataFrame, group_column: str, flag_column: str = "broad_conflict") -> pd.DataFrame:
    return (
        frame.groupby(group_column, dropna=False)
        .agg(
            n=(flag_column, "size"),
            conflict_files=(flag_column, "sum"),
            mean_range=("detector_score_range", "mean"),
            median_range=("detector_score_range", "median"),
        )
        .reset_index()
        .assign(conflict_percent=lambda table: 100 * table["conflict_files"] / table["n"])
        .sort_values("conflict_percent", ascending=False)
    )


def detector_low_high_patterns(frame: pd.DataFrame, flag_column: str = "high_impact_conflict") -> pd.DataFrame:
    subset = frame[frame[flag_column]].copy()
    rows: list[dict[str, object]] = []
    for tool in TOOLS:
        column = f"{tool}_ai_rate_percent"
        if column not in subset.columns:
            continue
        rows.append(
            {
                "detector": tool,
                "n_flagged": len(subset),
                "score_le_30": int(pd.to_numeric(subset[column], errors="coerce").le(30).sum()),
                "score_ge_70": int(pd.to_numeric(subset[column], errors="coerce").ge(70).sum()),
            }
        )
    return pd.DataFrame(rows)
