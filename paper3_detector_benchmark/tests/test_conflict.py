import pandas as pd

from paper3_benchmark.conflict import add_conflict_fields


def test_conflict_patterns() -> None:
    rows = [
        {
            "Language": "Arabic",
            "Label": "AI-Free",
            "file_name": "a.txt",
            "truth_class": "Human",
            "gptzero_pred_label": "Human",
            "pangram_pred_label": "Human",
            "sapling_pred_label": "AI",
            "isgen_pred_label": "Human",
            "gptzero_score_percent": 1,
            "pangram_score_percent": 2,
            "sapling_score_percent": 90,
            "isgen_score_percent": 3,
        },
        {
            "Language": "English",
            "Label": "Humanized AI",
            "file_name": "b.txt",
            "truth_class": "AI",
            "gptzero_pred_label": "AI",
            "pangram_pred_label": "AI",
            "sapling_pred_label": "Human",
            "isgen_pred_label": "Human",
            "gptzero_score_percent": 90,
            "pangram_score_percent": 80,
            "sapling_score_percent": 20,
            "isgen_score_percent": 10,
        },
    ]
    result = add_conflict_fields(pd.DataFrame(rows))
    assert result.loc[0, "pattern_type"] == "Lone dissenter"
    assert result.loc[0, "lone_dissenter_tool"] == "sapling"
    assert result.loc[1, "pattern_type"] == "2 vs 2 split"
    assert result.loc[1, "text_score_range"] == 80
