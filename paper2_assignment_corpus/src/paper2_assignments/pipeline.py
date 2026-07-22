from __future__ import annotations

from pathlib import Path

import pandas as pd

from .conflict import add_conflict_fields, conflict_by_group, conflict_summary, detector_low_high_patterns, pairwise_detector_differences
from .io_utils import standardize_result_columns
from .scoring import SCORE_COLUMNS, add_combined_scores
from .statistics import detector_language_comparisons, group_summary, kruskal_wallis, pairwise_mann_whitney, word_count_correlations
from .weights import DEFAULT_WEIGHTS

GROUP_COLUMNS = ["university", "major", "year_original", "language", "assignment_type"]


def run_primary_analysis(frame: pd.DataFrame, output_dir: str | Path, *, weights: dict[str, dict[str, float]] = DEFAULT_WEIGHTS) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = standardize_result_columns(frame)
    required = ["language", "university", "major", "assignment_type", "year_original", "file_name", "word_count"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing assignment-level columns: {missing}")
    data = add_combined_scores(data, weights)
    data = add_conflict_fields(data)

    descriptive_tables: list[pd.DataFrame] = []
    kruskal_rows: list[dict[str, object]] = []
    pairwise_tables: list[pd.DataFrame] = []
    for column in GROUP_COLUMNS:
        summary = group_summary(data, column)
        summary.insert(0, "grouping_variable", column)
        descriptive_tables.append(summary)
        kruskal_rows.append(kruskal_wallis(data, column))
        pairwise = pairwise_mann_whitney(data, column)
        if not pairwise.empty:
            pairwise_tables.append(pairwise)

    score_columns = ["combined_ai_rate_percent"] + [column for column in SCORE_COLUMNS if column in data.columns]
    outputs = {
        "assignment_level_analysis": data,
        "descriptive_groups": pd.concat(descriptive_tables, ignore_index=True),
        "kruskal_wallis": pd.DataFrame(kruskal_rows),
        "pairwise_posthoc": pd.concat(pairwise_tables, ignore_index=True) if pairwise_tables else pd.DataFrame(),
        "word_count_correlations": word_count_correlations(data, score_columns),
        "detector_language_comparisons": detector_language_comparisons(data),
        "conflict_summary": conflict_summary(data),
        "pairwise_detector_differences": pairwise_detector_differences(data),
        "conflict_by_language": conflict_by_group(data, "language"),
        "conflict_by_assignment_type": conflict_by_group(data, "assignment_type"),
        "high_impact_detector_patterns": detector_low_high_patterns(data),
    }
    with pd.ExcelWriter(output_dir / "paper2_primary_analysis_tables.xlsx", engine="openpyxl") as writer:
        for name, table in outputs.items():
            if name == "assignment_level_analysis":
                continue
            table.to_excel(writer, sheet_name=name[:31], index=False)
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return outputs
