from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests

from .schema import TEXT_TOOLS


def cochran_q_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for language in ("Arabic", "English"):
        subset = df[df["Language"] == language]
        columns = [f"{tool}_correct" for tool in TEXT_TOOLS]
        evaluable = subset.dropna(subset=columns).copy()
        matrix = evaluable[columns].astype(int).to_numpy()
        if len(evaluable) == 0:
            continue
        result = cochrans_q(matrix)
        rows.append(
            {
                "Language": language,
                "n": int(len(evaluable)),
                "Q_statistic": float(result.statistic),
                "df": len(TEXT_TOOLS) - 1,
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def exact_mcnemar_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for language in ("Arabic", "English"):
        subset = df[df["Language"] == language].copy()
        local: list[dict[str, object]] = []
        raw_p: list[float] = []
        for tool_a, tool_b in combinations(TEXT_TOOLS, 2):
            cols = [f"{tool_a}_correct", f"{tool_b}_correct"]
            pair = subset.dropna(subset=cols).copy()
            a = pair[cols[0]].astype(bool)
            b = pair[cols[1]].astype(bool)
            a_only = int((a & ~b).sum())
            b_only = int((~a & b).sum())
            discordant = a_only + b_only
            p_value = 1.0 if discordant == 0 else float(
                binomtest(min(a_only, b_only), n=discordant, p=0.5, alternative="two-sided").pvalue
            )
            raw_p.append(p_value)
            local.append(
                {
                    "Language": language,
                    "Detector_A": tool_a,
                    "Detector_B": tool_b,
                    "Accuracy_A_Percent": 100 * float(a.mean()),
                    "Accuracy_B_Percent": 100 * float(b.mean()),
                    "A_Correct_B_Incorrect": a_only,
                    "B_Correct_A_Incorrect": b_only,
                    "Discordant_Pairs": discordant,
                    "Exact_p_value": p_value,
                }
            )
        adjusted = multipletests(raw_p, method="fdr_bh")[1]
        for row, adjusted_p in zip(local, adjusted):
            row["BH_Adjusted_p_value"] = float(adjusted_p)
            if row["Accuracy_A_Percent"] > row["Accuracy_B_Percent"]:
                row["Higher_Accuracy_Detector"] = row["Detector_A"]
            elif row["Accuracy_B_Percent"] > row["Accuracy_A_Percent"]:
                row["Higher_Accuracy_Detector"] = row["Detector_B"]
            else:
                row["Higher_Accuracy_Detector"] = "Tie"
            row["Significant_After_BH"] = bool(adjusted_p < 0.05)
            rows.append(row)
    return pd.DataFrame(rows)
