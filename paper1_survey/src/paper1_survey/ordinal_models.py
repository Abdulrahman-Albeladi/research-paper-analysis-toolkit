from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .psychometrics import add_construct_scores, transform_contribution_to_five_point
from .schema import DEFAULT_SCHEMA, SurveySchema


@dataclass
class OrderedFit:
    name: str
    result: object
    n: int
    predictors: list[str]
    outcome: str


def fit_ordered_logit(outcome: pd.Series, predictors: pd.DataFrame):
    data = pd.concat([outcome.rename("__outcome__"), predictors], axis=1).dropna()
    y = pd.to_numeric(data.pop("__outcome__"), errors="coerce").round().astype(int)
    x = data.astype(float)
    if len(data) < 30 or y.nunique() < 3:
        raise ValueError("Ordered logit requires at least 30 complete rows and 3 outcome levels.")
    return OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False)


def fit_null_ordered_logit(outcome: pd.Series, index: pd.Index):
    y = pd.to_numeric(outcome.loc[index], errors="coerce").round().astype(int)
    x = pd.DataFrame({"_null": np.zeros(len(index))}, index=index)
    return OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False)


def likelihood_ratio_test(full_result, reduced_result, df_difference: int) -> tuple[float, int, float]:
    statistic = 2 * (float(full_result.llf) - float(reduced_result.llf))
    return statistic, int(df_difference), float(stats.chi2.sf(statistic, df_difference))


def mcfadden_r2(result, null_result) -> float:
    return float(1 - result.llf / null_result.llf)


def tidy_ordered_result(result, predictors: list[str], model: str, outcome: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in predictors:
        beta = float(result.params[term])
        standard_error = float(result.bse[term])
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "term": term,
                "beta": beta,
                "odds_ratio": float(np.exp(beta)),
                "ci_low": float(np.exp(beta - 1.96 * standard_error)),
                "ci_high": float(np.exp(beta + 1.96 * standard_error)),
                "p": float(result.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def proportional_odds_diagnostic(outcome: pd.Series, predictors: pd.DataFrame, min_per_side: int = 30) -> pd.DataFrame:
    data = pd.concat([outcome.rename("__outcome__"), predictors], axis=1).dropna()
    y = pd.to_numeric(data.pop("__outcome__"), errors="coerce").round().astype(int)
    x = data.astype(float)
    rows: list[dict[str, object]] = []
    levels = sorted(y.unique())
    for cutoff in levels[:-1]:
        binary = (y > cutoff).astype(int)
        if binary.sum() < min_per_side or (1 - binary).sum() < min_per_side:
            continue
        result = sm.Logit(binary, sm.add_constant(x, has_constant="add")).fit(disp=False)
        for term in x.columns:
            rows.append(
                {
                    "cutoff": cutoff,
                    "term": term,
                    "beta": float(result.params[term]),
                    "odds_ratio": float(np.exp(result.params[term])),
                    "p": float(result.pvalues[term]),
                }
            )
    return pd.DataFrame(rows)


def _prepare_composite_predictors(frame: pd.DataFrame, schema: SurveySchema) -> pd.DataFrame:
    scored = add_construct_scores(frame, schema)
    scored["AI_Contribution_Five_Point"] = transform_contribution_to_five_point(scored[schema.contribution])
    return scored


def run_composite_models(frame: pd.DataFrame, schema: SurveySchema = DEFAULT_SCHEMA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _prepare_composite_predictors(frame, schema)
    base_predictors = [schema.frequency, "AI_Contribution_Five_Point", "AI_Task_Support_Score", "AI_Skills_Support_Score"]
    full_predictors = base_predictors + ["AI_Skills_Impact_Score"]
    outcomes = [schema.mastery, schema.gpa, schema.confidence]

    coefficient_tables: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    po_tables: list[pd.DataFrame] = []

    for outcome in outcomes:
        complete = data[[outcome] + full_predictors].apply(pd.to_numeric, errors="coerce").dropna()
        if len(complete) < 80 or complete[outcome].round().nunique() < 3:
            fit_rows.append({"outcome": outcome, "status": "insufficient complete cases"})
            continue
        y = complete[outcome].round().astype(int)
        x_full = complete[full_predictors]
        x_base = complete[base_predictors]
        x_null = pd.DataFrame({"_null": np.zeros(len(complete))}, index=complete.index)
        full = OrderedModel(y, x_full, distr="logit").fit(method="bfgs", disp=False)
        base = OrderedModel(y, x_base, distr="logit").fit(method="bfgs", disp=False)
        null = OrderedModel(y, x_null, distr="logit").fit(method="bfgs", disp=False)
        full_lrt = likelihood_ratio_test(full, null, len(full_predictors))
        ablation_lrt = likelihood_ratio_test(full, base, 1)
        fit_rows.append(
            {
                "outcome": outcome,
                "n": len(complete),
                "mcfadden_r2_full": mcfadden_r2(full, null),
                "mcfadden_r2_base": mcfadden_r2(base, null),
                "full_lrt_chi2": full_lrt[0],
                "full_lrt_df": full_lrt[1],
                "full_lrt_p": full_lrt[2],
                "skills_impact_ablation_chi2": ablation_lrt[0],
                "skills_impact_ablation_df": ablation_lrt[1],
                "skills_impact_ablation_p": ablation_lrt[2],
                "status": "ok",
            }
        )
        coefficient_tables.append(tidy_ordered_result(full, full_predictors, "full_composite", outcome))
        po = proportional_odds_diagnostic(y, x_full)
        if not po.empty:
            po.insert(0, "outcome", outcome)
            po_tables.append(po)
    return (
        pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else pd.DataFrame(),
        pd.DataFrame(fit_rows),
        pd.concat(po_tables, ignore_index=True) if po_tables else pd.DataFrame(),
    )


def screen_categorical_predictor(frame: pd.DataFrame, outcome: str, predictor: str, min_group_n: int = 30) -> tuple[pd.DataFrame, dict[str, object]]:
    data = frame[[outcome, predictor]].dropna().copy()
    counts = data[predictor].value_counts()
    keep = counts[counts >= min_group_n].index
    data = data[data[predictor].isin(keep)]
    if len(keep) < 2:
        return pd.DataFrame(), {"outcome": outcome, "predictor": predictor, "status": "fewer than two eligible groups"}
    y = pd.to_numeric(data[outcome], errors="coerce").round()
    x = pd.get_dummies(data[predictor].astype(str), drop_first=True, dtype=float)
    complete = pd.concat([y.rename("y"), x], axis=1).dropna()
    y = complete.pop("y").astype(int)
    x = complete.astype(float)
    full = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False)
    null_x = pd.DataFrame({"_null": np.zeros(len(y))}, index=y.index)
    null = OrderedModel(y, null_x, distr="logit").fit(method="bfgs", disp=False)
    statistic, degrees, p_value = likelihood_ratio_test(full, null, x.shape[1])
    table = tidy_ordered_result(full, list(x.columns), f"screen_{predictor}", outcome)
    return table, {
        "outcome": outcome,
        "predictor": predictor,
        "n": len(y),
        "groups": len(keep),
        "mcfadden_r2": mcfadden_r2(full, null),
        "lrt_chi2": statistic,
        "lrt_df": degrees,
        "lrt_p": p_value,
        "status": "ok",
    }
