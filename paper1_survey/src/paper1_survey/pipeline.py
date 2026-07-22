from __future__ import annotations

from pathlib import Path

import pandas as pd

from .descriptives import numeric_summary, weighted_value_counts
from .io import read_table
from .ordinal_models import run_composite_models, screen_categorical_predictor
from .psychometrics import psychometric_summary
from .schema import DEFAULT_SCHEMA, SurveySchema


def run_main_analysis(frame: pd.DataFrame, output_dir: str | Path, schema: SurveySchema = DEFAULT_SCHEMA) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored, reliability, loadings = psychometric_summary(frame, schema)
    coefficients, model_fit, po_diagnostics = run_composite_models(scored, schema)

    categorical_tables: list[pd.DataFrame] = []
    categorical_fit: list[dict[str, object]] = []
    screen_predictors = [schema.gender, "الجامعة_المجمعة", "التخصص_المجموع"]
    for outcome in (schema.frequency, schema.contribution, schema.mastery, schema.gpa, schema.confidence):
        if outcome not in scored.columns:
            continue
        for predictor in screen_predictors:
            if predictor not in scored.columns:
                continue
            try:
                table, fit = screen_categorical_predictor(scored, outcome, predictor)
            except Exception as error:
                table, fit = pd.DataFrame(), {"outcome": outcome, "predictor": predictor, "status": repr(error)}
            if not table.empty:
                categorical_tables.append(table)
            categorical_fit.append(fit)

    numeric_columns = [schema.frequency, schema.contribution, schema.mastery, schema.gpa, schema.confidence]
    numeric = numeric_summary(scored, numeric_columns)
    category_outputs: list[pd.DataFrame] = []
    for column in (schema.gender, "الجامعة_المجمعة", "التخصص_المجموع", schema.ai_use, schema.frequency, schema.policy_preference):
        if column not in scored.columns:
            continue
        table = weighted_value_counts(scored[column], scored.get("analysis_weight", pd.Series(1.0, index=scored.index)))
        table.insert(0, "variable", column)
        category_outputs.append(table)
    categories = pd.concat(category_outputs, ignore_index=True) if category_outputs else pd.DataFrame()

    outputs = {
        "scored_data": scored,
        "psychometric_summary": reliability,
        "factor_loadings": loadings,
        "numeric_descriptives": numeric,
        "categorical_descriptives": categories,
        "composite_model_fit": model_fit,
        "composite_model_coefficients": coefficients,
        "proportional_odds_diagnostics": po_diagnostics,
        "categorical_screen_fit": pd.DataFrame(categorical_fit),
        "categorical_screen_coefficients": pd.concat(categorical_tables, ignore_index=True) if categorical_tables else pd.DataFrame(),
    }

    workbook = output_dir / "paper1_analysis_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, table in outputs.items():
            if name == "scored_data":
                continue
            table.to_excel(writer, sheet_name=name[:31], index=False)
    for name, table in outputs.items():
        if name != "scored_data":
            table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(output_dir / "survey_with_construct_scores.csv", index=False, encoding="utf-8-sig")
    return outputs


def run_main_analysis_from_path(input_path: str | Path, output_dir: str | Path, schema: SurveySchema = DEFAULT_SCHEMA) -> dict[str, pd.DataFrame]:
    return run_main_analysis(read_table(input_path), output_dir, schema)
