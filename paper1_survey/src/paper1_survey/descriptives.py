from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_value_counts(series: pd.Series, weights: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"value": series, "weight": pd.to_numeric(weights, errors="coerce")}).dropna()
    if data.empty:
        return pd.DataFrame(columns=["value", "unweighted_n", "unweighted_percent", "weighted_n", "weighted_percent"])
    unweighted = data["value"].value_counts(dropna=False).rename("unweighted_n")
    weighted = data.groupby("value", dropna=False)["weight"].sum().rename("weighted_n")
    out = pd.concat([unweighted, weighted], axis=1).fillna(0).reset_index(names="value")
    out["unweighted_percent"] = 100 * out["unweighted_n"] / out["unweighted_n"].sum()
    out["weighted_percent"] = 100 * out["weighted_n"] / out["weighted_n"].sum()
    return out[["value", "unweighted_n", "unweighted_percent", "weighted_n", "weighted_percent"]]


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weight_values.notna() & (weight_values > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weight_values[valid]))


def numeric_summary(frame: pd.DataFrame, columns: list[str], weight_column: str = "analysis_weight") -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "n": int(values.notna().sum()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "weighted_mean": weighted_mean(values, frame[weight_column]) if weight_column in frame.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)
