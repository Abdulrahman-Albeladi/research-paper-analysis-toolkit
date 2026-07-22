import numpy as np
import pandas as pd

from paper1_survey.psychometrics import cronbach_alpha, spearman_brown_two_item


def test_reliability_helpers():
    frame = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]})
    assert np.isclose(cronbach_alpha(frame), 1.0)
    assert np.isclose(spearman_brown_two_item(frame[["a", "b"]]), 1.0)
