import pandas as pd

from paper2_assignments.conflict import add_conflict_fields


def test_high_impact_conflict_rule():
    frame = pd.DataFrame(
        {
            "combined_ai_rate_percent": [50, 80],
            "gptzero_ai_rate_percent": [20, 75],
            "pangram_ai_rate_percent": [80, 85],
            "sapling_ai_rate_percent": [50, 80],
            "isgen_ai_rate_percent": [55, 78],
        }
    )
    out = add_conflict_fields(frame)
    assert bool(out.loc[0, "high_impact_conflict"])
    assert not bool(out.loc[1, "high_impact_conflict"])
