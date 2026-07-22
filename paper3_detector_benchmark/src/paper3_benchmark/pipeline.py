from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .conflict import (
    add_conflict_fields,
    coding_accuracy_by_label,
    conflict_counts,
    dominant_conflict_signatures,
    pattern_counts,
    uncertainty_tests,
)
from .features import build_feature_table, compare_structural_features
from .io_utils import load_manifest, read_table, write_csv, write_workbook
from .metrics import (
    compute_metrics_by_label,
    compute_overall_metrics,
    dataset_audit,
    mean_word_count_table,
    paper_table_1,
    paper_table_2,
    paper_table_3a,
    paper_table_3b,
    standardize_results,
)
from .statistics import cochran_q_tests, exact_mcnemar_posthoc
from .weights import compute_methodological_weights


def run_analysis(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    files_root: str | Path | None = None,
    sheet_name: str | int | None = 0,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_table(results_path, sheet_name=sheet_name)
    standardized = standardize_results(raw)
    conflict_master = add_conflict_fields(standardized)

    overall = compute_overall_metrics(standardized)
    by_label = compute_metrics_by_label(standardized)
    tables: dict[str, pd.DataFrame] = {
        "dataset_audit": dataset_audit(standardized),
        "analysis_master": conflict_master,
        "word_count_summary": mean_word_count_table(standardized),
        "paper_table_1": paper_table_1(standardized),
        "overall_metrics": overall,
        "paper_table_2": paper_table_2(overall),
        "metrics_by_label": by_label,
        "paper_table_3a": paper_table_3a(by_label),
        "paper_table_3b": paper_table_3b(by_label),
        "methodological_weights": compute_methodological_weights(by_label),
        "cochrans_q": cochran_q_tests(standardized),
        "mcnemar_posthoc": exact_mcnemar_posthoc(standardized),
        "conflict_counts": conflict_counts(conflict_master),
        "pattern_counts": pattern_counts(conflict_master),
        "dominant_signatures": dominant_conflict_signatures(conflict_master),
        "uncertainty_tests": uncertainty_tests(conflict_master),
        "coding_accuracy": coding_accuracy_by_label(conflict_master),
    }

    if manifest_path is not None:
        manifest = load_manifest(manifest_path, root=files_root)
        features = build_feature_table(manifest)
        merged = conflict_master.merge(
            features,
            left_on=["Language", "Label", "file_name"],
            right_on=["Language", "Label", "file_name"],
            how="left",
        )
        tables["feature_table"] = features
        tables["master_with_features"] = merged
        tables["feature_comparisons"] = compare_structural_features(merged)

    for name, table in tables.items():
        write_csv(table, output_dir / f"{name}.csv")
    write_workbook(tables, output_dir / "paper3_analysis_outputs.xlsx")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_path": str(Path(results_path)),
        "manifest_path": str(Path(manifest_path)) if manifest_path else None,
        "n_rows": int(len(standardized)),
        "analysis_threshold_percent": 50,
        "software": "paper3-ai-detector-benchmark 1.0.0",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tables
