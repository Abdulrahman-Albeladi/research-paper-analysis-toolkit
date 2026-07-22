import math

import pandas as pd

from paper3_benchmark.weights import compute_methodological_weights


def test_methodological_weight_formula_and_normalization() -> None:
    rows = []
    for language in ("Arabic", "English"):
        for tool, values in {
            "gptzero": (100, 100, 25),
            "pangram": (100, 100, 64),
            "sapling": (10, 78, 70),
            "isgen": (96, 72, 78),
        }.items():
            for label, value in zip(("AI-Free", "AI-Generated", "Humanized AI"), values):
                rows.append(
                    {
                        "Language": language,
                        "Label": label,
                        "Tool": tool,
                        "Soft_Accuracy_Percent": float(value),
                        "Binary_Accuracy_Percent": float(value),
                    }
                )
    result = compute_methodological_weights(pd.DataFrame(rows))
    arabic = result[result["Language"] == "Arabic"]
    assert math.isclose(arabic["Normalized_Soft_Weight"].sum(), 1.0)
    pangram = arabic[arabic["Tool"] == "pangram"].iloc[0]
    expected = 100 * (1.0**0.4) * (1.0**0.4) * (0.64**0.2)
    assert math.isclose(pangram["Methodological_Soft_Score"], expected)
