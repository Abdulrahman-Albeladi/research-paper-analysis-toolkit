from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LABELS, TEXT_TOOLS


def _geometric_method_score(human: float, generated: float, humanized: float) -> float:
    values = np.array([human, generated, humanized], dtype=float) / 100.0
    if np.isnan(values).any() or (values < 0).any():
        return np.nan
    return float((values[0] ** 0.4) * (values[1] ** 0.4) * (values[2] ** 0.2) * 100.0)


def compute_methodological_weights(by_label: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for language in ("Arabic", "English"):
        local_rows: list[dict[str, object]] = []
        for tool in TEXT_TOOLS:
            subset = by_label[(by_label["Language"] == language) & (by_label["Tool"] == tool)]
            lookup = subset.set_index("Label")
            soft = {
                label: float(lookup.loc[label, "Soft_Accuracy_Percent"]) if label in lookup.index else np.nan
                for label in LABELS
            }
            binary = {
                label: float(lookup.loc[label, "Binary_Accuracy_Percent"]) if label in lookup.index else np.nan
                for label in LABELS
            }
            local_rows.append(
                {
                    "Language": language,
                    "Tool": tool,
                    "AI_Free_Soft_Accuracy_Percent": soft["AI-Free"],
                    "AI_Generated_Soft_Accuracy_Percent": soft["AI-Generated"],
                    "Humanized_AI_Soft_Accuracy_Percent": soft["Humanized AI"],
                    "Methodological_Soft_Score": _geometric_method_score(
                        soft["AI-Free"], soft["AI-Generated"], soft["Humanized AI"]
                    ),
                    "AI_Free_Binary_Accuracy_Percent": binary["AI-Free"],
                    "AI_Generated_Binary_Accuracy_Percent": binary["AI-Generated"],
                    "Humanized_AI_Binary_Accuracy_Percent": binary["Humanized AI"],
                    "Methodological_Binary_Score": _geometric_method_score(
                        binary["AI-Free"], binary["AI-Generated"], binary["Humanized AI"]
                    ),
                }
            )
        soft_total = sum(row["Methodological_Soft_Score"] for row in local_rows if pd.notna(row["Methodological_Soft_Score"]))
        binary_total = sum(row["Methodological_Binary_Score"] for row in local_rows if pd.notna(row["Methodological_Binary_Score"]))
        for row in local_rows:
            row["Normalized_Soft_Weight"] = row["Methodological_Soft_Score"] / soft_total if soft_total else np.nan
            row["Normalized_Binary_Weight"] = row["Methodological_Binary_Score"] / binary_total if binary_total else np.nan
            rows.append(row)

    rows.append(
        {
            "Language": "Coding",
            "Tool": "pangram",
            "AI_Free_Soft_Accuracy_Percent": np.nan,
            "AI_Generated_Soft_Accuracy_Percent": np.nan,
            "Humanized_AI_Soft_Accuracy_Percent": np.nan,
            "Methodological_Soft_Score": np.nan,
            "AI_Free_Binary_Accuracy_Percent": np.nan,
            "AI_Generated_Binary_Accuracy_Percent": np.nan,
            "Humanized_AI_Binary_Accuracy_Percent": np.nan,
            "Methodological_Binary_Score": np.nan,
            "Normalized_Soft_Weight": 1.0,
            "Normalized_Binary_Weight": 1.0,
        }
    )
    return pd.DataFrame(rows)
