import numpy as np
import pandas as pd

from paper2_assignments.scoring import add_combined_scores, combined_ai_rate


def test_language_specific_weighting():
    row = pd.Series({"language": "Arabic", "gptzero_ai_rate_percent": 100, "pangram_ai_rate_percent": 0, "sapling_ai_rate_percent": 0, "isgen_ai_rate_percent": 0})
    assert np.isclose(combined_ai_rate(row), 24.0)


def test_code_uses_pangram_only():
    frame = pd.DataFrame({"language": ["Code"], "pangram_ai_rate_percent": [55], "gptzero_ai_rate_percent": [0]})
    out = add_combined_scores(frame)
    assert out.loc[0, "combined_ai_rate_percent"] == 55
