import math

import pandas as pd

from paper3_benchmark.metrics import compute_metrics_by_label, compute_overall_metrics, standardize_results


def synthetic_results() -> pd.DataFrame:
    rows = []
    for language in ("Arabic", "English"):
        for label, score in (("AI-Free", 10), ("AI-Generated", 90), ("Humanized AI", 70)):
            row = {"Language": language, "Label": label, "file_name": f"{language}_{label}.txt", "word_count": 100}
            for tool in ("gptzero", "pangram", "sapling", "isgen"):
                row[f"{tool}_ai_rate_percent"] = score
                row[f"{tool}_result"] = "Likely AI" if score >= 50 else "Likely Human"
            rows.append(row)
    for label, score in (("AI-Free", 0), ("AI-Generated", 80), ("Humanized AI", 20)):
        rows.append(
            {
                "Language": "Coding",
                "Label": label,
                "file_name": f"Coding_{label}.txt",
                "word_count": 100,
                "pangram_ai_rate_percent": score,
                "pangram_result": "Likely AI" if score >= 50 else "Likely Human",
            }
        )
    return pd.DataFrame(rows)


def test_standardization_and_metrics() -> None:
    standardized = standardize_results(synthetic_results())
    overall = compute_overall_metrics(standardized)
    arabic_gptzero = overall[(overall["Language"] == "Arabic") & (overall["Tool"] == "gptzero")].iloc[0]
    assert arabic_gptzero["Binary_Accuracy_Percent"] == 100.0
    assert math.isclose(arabic_gptzero["Soft_Accuracy_Percent"], (90 + 90 + 70) / 3)


def test_coding_uses_pangram_only() -> None:
    standardized = standardize_results(synthetic_results())
    overall = compute_overall_metrics(standardized)
    coding = overall[overall["Language"] == "Coding"]
    assert coding["Tool"].tolist() == ["pangram"]


def test_metrics_by_label_has_all_conditions() -> None:
    standardized = standardize_results(synthetic_results())
    table = compute_metrics_by_label(standardized)
    assert set(table[table["Language"] == "Arabic"]["Label"]) == {"AI-Free", "AI-Generated", "Humanized AI"}
