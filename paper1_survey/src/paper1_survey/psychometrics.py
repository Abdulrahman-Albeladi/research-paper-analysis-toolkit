from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .schema import DEFAULT_SCHEMA, SurveySchema, construct_definitions


def numeric_complete(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").dropna()


def cronbach_alpha(frame: pd.DataFrame) -> float:
    x = frame.dropna().astype(float)
    n_items = x.shape[1]
    if n_items < 2 or len(x) < 2:
        return float("nan")
    total_variance = x.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return float("nan")
    item_variance = x.var(axis=0, ddof=1).sum()
    return float((n_items / (n_items - 1)) * (1 - item_variance / total_variance))


def spearman_brown_two_item(frame: pd.DataFrame) -> float:
    x = frame.dropna().astype(float)
    if x.shape[1] != 2 or len(x) < 3:
        return float("nan")
    correlation = x.iloc[:, 0].corr(x.iloc[:, 1])
    if pd.isna(correlation) or np.isclose(correlation, -1):
        return float("nan")
    return float((2 * correlation) / (1 + correlation))


def kmo_bartlett(frame: pd.DataFrame) -> tuple[float, float, float]:
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    x = frame.dropna().astype(float)
    _, kmo_model = calculate_kmo(x)
    chi_square, p_value = calculate_bartlett_sphericity(x)
    return float(kmo_model), float(chi_square), float(p_value)


def one_factor_efa(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    from factor_analyzer import FactorAnalyzer

    x = frame.dropna().astype(float)
    analyzer = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    analyzer.fit(x)
    loadings = analyzer.loadings_.flatten().astype(float)
    variance_explained = float(analyzer.get_factor_variance()[1][0] * 100)
    return loadings, variance_explained


def transform_contribution_to_five_point(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (1 + (values - 1) * (4 / 3)).clip(1, 5)


def add_construct_scores(frame: pd.DataFrame, schema: SurveySchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    out = frame.copy()
    definitions = construct_definitions(schema)
    if schema.contribution in out.columns:
        out[f"{schema.contribution}__five_point"] = transform_contribution_to_five_point(out[schema.contribution])
        definitions["AI_Use_Intensity"] = [schema.frequency, f"{schema.contribution}__five_point"]
    for construct, columns in definitions.items():
        available = [column for column in columns if column in out.columns]
        if len(available) != len(columns):
            continue
        out[f"{construct}_Score"] = out[available].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out


def psychometric_summary(frame: pd.DataFrame, schema: SurveySchema = DEFAULT_SCHEMA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = add_construct_scores(frame, schema)
    definitions = construct_definitions(schema)
    if f"{schema.contribution}__five_point" in scored.columns:
        definitions["AI_Use_Intensity"] = [schema.frequency, f"{schema.contribution}__five_point"]

    summary_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    for construct, columns in definitions.items():
        if any(column not in scored.columns for column in columns):
            summary_rows.append({"construct": construct, "items": len(columns), "status": "missing columns"})
            continue
        block = numeric_complete(scored, columns)
        row: dict[str, object] = {
            "construct": construct,
            "items": len(columns),
            "n_complete": len(block),
            "cronbach_alpha": cronbach_alpha(block),
            "score_mean": scored[f"{construct}_Score"].mean(),
            "score_sd": scored[f"{construct}_Score"].std(ddof=1),
            "status": "ok",
        }
        if len(columns) == 2:
            row["spearman_brown"] = spearman_brown_two_item(block)
            row["kmo"] = np.nan
            row["bartlett_chi2"] = np.nan
            row["bartlett_p"] = np.nan
            row["factor1_variance_percent"] = np.nan
            row["loading_min"] = np.nan
            row["loading_max"] = np.nan
        elif len(block) >= max(20, len(columns) * 3):
            kmo, bartlett_chi2, bartlett_p = kmo_bartlett(block)
            loadings, variance = one_factor_efa(block)
            row.update(
                {
                    "spearman_brown": np.nan,
                    "kmo": kmo,
                    "bartlett_chi2": bartlett_chi2,
                    "bartlett_p": bartlett_p,
                    "factor1_variance_percent": variance,
                    "loading_min": float(np.min(loadings)),
                    "loading_max": float(np.max(loadings)),
                }
            )
            for item, loading in zip(columns, loadings):
                loading_rows.append({"construct": construct, "item": item, "factor1_loading": float(loading)})
        else:
            row["status"] = "insufficient complete cases for EFA"
        summary_rows.append(row)
    return scored, pd.DataFrame(summary_rows), pd.DataFrame(loading_rows)


def pilot_scale_diagnostics(frame: pd.DataFrame, scale_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    for scale, columns in scale_map.items():
        available = [column for column in columns if column in frame.columns]
        if len(available) != len(columns):
            summary_rows.append({"scale": scale, "items": len(columns), "status": "missing columns"})
            continue
        block = numeric_complete(frame, columns)
        row: dict[str, object] = {
            "scale": scale,
            "items": len(columns),
            "n_complete": len(block),
            "alpha": cronbach_alpha(block),
            "status": "ok",
        }
        if len(columns) == 2:
            row["spearman_brown"] = spearman_brown_two_item(block)
        elif len(block) >= max(20, len(columns) * 3):
            kmo, chi_square, p_value = kmo_bartlett(block)
            loadings, variance = one_factor_efa(block)
            row.update({"kmo": kmo, "bartlett_chi2": chi_square, "bartlett_p": p_value, "variance_percent": variance})
            for item, loading in zip(columns, loadings):
                loading_rows.append({"scale": scale, "item": item, "loading": float(loading)})
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(loading_rows)
