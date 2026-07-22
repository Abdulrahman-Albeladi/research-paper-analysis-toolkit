from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


def collapse_year(row: pd.Series) -> str | float:
    original = str(row.get("year_original", "")).strip()
    major = str(row.get("major", "")).strip()
    assignment_type = str(row.get("assignment_type", "")).strip()
    if original == "Medical" and major == "Med":
        if assignment_type in {"OSCE", "Manuscripts"}:
            return "First Year"
        if assignment_type == "Research Projects":
            return "Second Year"
        return np.nan
    if "First Year" in original:
        return "First Year"
    if "Second Year" in original:
        return "Second Year"
    return original or np.nan


def assignment_context(row: pd.Series) -> str:
    language = str(row.get("language", ""))
    assignment_type = str(row.get("assignment_type", ""))
    major = str(row.get("major", ""))
    if language == "Arabic" and assignment_type == "Article":
        return "Arabic Article / General"
    if assignment_type == "Project Report" and major == "MechEng":
        return "Engineering Project Report"
    return assignment_type


def prepare_adjusted_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["combined_ai_rate_percent", "language", "assignment_type", "word_count", "major", "university", "year_original"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing adjusted-analysis columns: {missing}")
    out = frame.copy()
    out["combined_ai_rate_percent"] = pd.to_numeric(out["combined_ai_rate_percent"], errors="coerce").clip(0, 100)
    out["word_count"] = pd.to_numeric(out["word_count"], errors="coerce")
    out = out.dropna(subset=required).copy()
    out["combined_ai_rate_prop"] = out["combined_ai_rate_percent"] / 100
    out["log_word_count"] = np.log1p(out["word_count"])
    out["year_grouping"] = out.apply(collapse_year, axis=1)
    out["assignment_context"] = out.apply(assignment_context, axis=1)
    return out


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi_square, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n <= 1:
        return np.nan, float(p_value)
    rows, columns = table.shape
    phi2 = chi_square / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    return (float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else np.nan, float(p_value))


def categorical_overlap_diagnostics(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or ["language", "assignment_type", "major", "university", "year_grouping"]
    rows: list[dict[str, object]] = []
    for first, second in combinations(columns, 2):
        table = pd.crosstab(frame[first], frame[second])
        value, p_value = cramers_v(table)
        row_proportions = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
        column_proportions = table.div(table.sum(axis=0).replace(0, np.nan), axis=1)
        rows.append(
            {
                "variable_1": first,
                "variable_2": second,
                "cramers_v": value,
                "chi_square_p": p_value,
                "max_row_purity": float(row_proportions.max(axis=1).max()),
                "max_column_purity": float(column_proportions.max(axis=0).max()),
                "zero_cells": int((table == 0).sum().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def compute_vif(frame: pd.DataFrame, categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    pieces = [frame[numeric].astype(float)] if numeric else []
    pieces.extend(pd.get_dummies(frame[column].astype(str), prefix=column, drop_first=True, dtype=float) for column in categorical)
    x = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    x = x.loc[:, x.nunique(dropna=False) > 1]
    x_constant = sm.add_constant(x, has_constant="add")
    rows: list[dict[str, object]] = []
    for index, term in enumerate(x_constant.columns):
        if term == "const":
            continue
        try:
            value = variance_inflation_factor(x_constant.values, index)
        except Exception:
            value = np.inf
        rows.append({"term": term, "vif": float(value)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def _fit_ols(formula: str, frame: pd.DataFrame, name: str):
    result = smf.ols(formula, data=frame).fit(cov_type="HC3")
    result.model_name = name
    result.model_family = "OLS-HC3"
    return result


def _fit_fractional_logit(formula: str, frame: pd.DataFrame, name: str):
    try:
        result = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(cov_type="HC3")
    except Exception:
        result = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(cov_type="HC0")
    result.model_name = name
    result.model_family = "Fractional logit GLM"
    return result


def tidy_model(result) -> pd.DataFrame:
    confidence = result.conf_int()
    rows: list[dict[str, object]] = []
    for term in result.params.index:
        if term == "Intercept":
            continue
        rows.append(
            {
                "model": result.model_name,
                "family": result.model_family,
                "term": term,
                "estimate": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "ci_low": float(confidence.loc[term, 0]),
                "ci_high": float(confidence.loc[term, 1]),
                "statistic": float(result.params[term] / result.bse[term]),
                "p": float(result.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def fit_summary(result) -> dict[str, object]:
    row: dict[str, object] = {"model": result.model_name, "family": result.model_family, "n": int(result.nobs), "aic": float(result.aic)}
    if hasattr(result, "rsquared_adj"):
        row.update({"r2": float(result.rsquared), "adjusted_r2": float(result.rsquared_adj)})
    else:
        try:
            row["mcfadden_pseudo_r2"] = float(1 - result.llf / result.llnull)
        except Exception:
            row["mcfadden_pseudo_r2"] = np.nan
    return row


def run_adjusted_models(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = prepare_adjusted_data(frame)
    text_data = data[data["language"] != "Code"].copy()
    formulas = [
        ("Full adjusted OLS-HC3", _fit_ols, "combined_ai_rate_percent ~ log_word_count + C(language) + C(assignment_type) + C(major) + C(university) + C(year_grouping)", data),
        ("Full adjusted fractional logit", _fit_fractional_logit, "combined_ai_rate_prop ~ log_word_count + C(language) + C(assignment_type) + C(major) + C(university) + C(year_grouping)", data),
        ("Reduced assignment-context OLS-HC3", _fit_ols, "combined_ai_rate_percent ~ log_word_count + C(assignment_context)", data),
        ("Reduced assignment-context fractional logit", _fit_fractional_logit, "combined_ai_rate_prop ~ log_word_count + C(assignment_context)", data),
        ("Text-only reduced assignment-context OLS-HC3", _fit_ols, "combined_ai_rate_percent ~ log_word_count + C(assignment_context)", text_data),
        ("Text-only reduced assignment-context fractional logit", _fit_fractional_logit, "combined_ai_rate_prop ~ log_word_count + C(assignment_context)", text_data),
    ]
    models = []
    errors: list[dict[str, object]] = []
    for name, fitting_function, formula, subset in formulas:
        try:
            models.append(fitting_function(formula, subset, name))
        except Exception as error:
            errors.append({"model": name, "formula": formula, "error": repr(error)})

    coefficients = pd.concat([tidy_model(model) for model in models], ignore_index=True) if models else pd.DataFrame()
    fit = pd.DataFrame([fit_summary(model) for model in models])
    overlap = categorical_overlap_diagnostics(data)
    vif_full = compute_vif(data, ["language", "assignment_type", "major", "university", "year_grouping"], ["log_word_count"])
    vif_context = compute_vif(data, ["assignment_context"], ["log_word_count"])
    contexts = data.groupby("assignment_context")["combined_ai_rate_percent"].agg(n="count", mean="mean", sd="std").reset_index().sort_values("mean", ascending=False)
    return {
        "analytic_data": data,
        "model_fit": fit,
        "model_coefficients": coefficients,
        "overlap_diagnostics": overlap,
        "vif_full": vif_full,
        "vif_context": vif_context,
        "assignment_context_descriptives": contexts,
        "model_errors": pd.DataFrame(errors),
    }
