import pandas as pd

from paper3_benchmark.metrics import standardize_results
from paper3_benchmark.statistics import cochran_q_tests, exact_mcnemar_posthoc


def test_paired_tests_return_expected_shapes() -> None:
    rows = []
    for language in ("Arabic", "English"):
        for i in range(12):
            label = "AI-Free" if i < 4 else ("AI-Generated" if i < 8 else "Humanized AI")
            truth_score = 10 if label == "AI-Free" else 90
            row = {"Language": language, "Label": label, "file_name": f"{language}_{i}.txt"}
            row["gptzero_ai_rate_percent"] = truth_score
            row["pangram_ai_rate_percent"] = truth_score
            row["sapling_ai_rate_percent"] = 90
            row["isgen_ai_rate_percent"] = 10
            rows.append(row)
    standardized = standardize_results(pd.DataFrame(rows))
    q = cochran_q_tests(standardized)
    mc = exact_mcnemar_posthoc(standardized)
    assert len(q) == 2
    assert len(mc) == 12
    assert set(mc["Language"]) == {"Arabic", "English"}
