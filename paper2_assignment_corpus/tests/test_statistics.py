import numpy as np
import pandas as pd

from paper2_assignments.statistics import cliffs_delta, kruskal_wallis, pairwise_mann_whitney


def test_nonparametric_helpers():
    frame = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "combined_ai_rate_percent": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]})
    result = kruskal_wallis(frame, "group")
    assert result["groups_tested"] == 2
    pairwise = pairwise_mann_whitney(frame, "group")
    assert len(pairwise) == 1
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
