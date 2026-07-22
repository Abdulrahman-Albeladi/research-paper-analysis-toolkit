from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import (
    DEFAULT_SCHEMA,
    SKILLS_IMPACT_ITEMS,
    SKILLS_SUPPORT_ITEMS,
    TASK_SUPPORT_ITEMS,
    SurveySchema,
    ensure_columns,
)

YES_NO_MAP = {"نعم": 1, "أحيانًا": 1, "قليلًا": 1, "لا": 0}
FREQUENCY_MAP = {"يوميًّا": 5, "يوميًّا": 5, "أسبوعيًّا": 4, "أسبوعيًّا": 4, "شهريًّا": 3, "شهريًّا": 3, "نادرًا": 2, "لا أستخدمه": 1}
CONTRIBUTION_MAP = {"أعتمد عليه بشكل كبير": 4, "يساهم بشكل معقول": 3, "يساهم قليلًا": 2, "لا أستخدمه": 1}
HELP_MAP = {"بشكل كبير جدًّا": 5, "بشكل كبير جدًا": 5, "كثيرًا": 4, "بشكل معقول": 3, "قليلًا": 2, "لا تساعدني": 1, "لا تساعد": 1}
IMPACT_MAP = {"بشكل إيجابي": 5, "إيجابي طفيف": 4, "لا يؤثّر": 3, "سلبي طفيف": 2, "بشكل سلبي": 1}

TOP_UNIVERSITIES = {
    "جامعة الطائف",
    "جامعة طيبة",
    "جامعة جدة",
    "جامعة الملك فهد للبترول والمعادن",
}

MAJOR_GROUPS = {
    "هندسة": {
        "الهندسة", "الهندسة الصناعية", "الهندسة المدنية", "الهندسة الكهربائية",
        "الهندسة الميكانيكية", "الهندسة المعمارية", "الهندسة البترولية",
        "الهندسة الكيميائية", "هندسة البرمجيات", "علوم وهندسة المواد",
    },
    "تقنية وبيانات": {"نظم المعلومات", "ذكاء اصطناعي", "تقنية المعلومات", "علوم الحاسب"},
    "علوم طبيعية ورياضيات": {"الفيزياء", "الكيمياء", "الأحياء", "الرياضيات والإحصاء", "كلية العلوم"},
    "صحة": {"الطب", "التمريض", "التغذية", "التغذية السريرية", "التقنية الحيوية"},
    "علوم اجتماعية": {"علم النفس", "الطفولة المبكرة", "إدارة الجودة", "الأنظمة", "الشريعة", "اللغة العربية", "اللغة الإنجليزية"},
    "إدارة وأعمال": {"إدارة الأعمال", "المحاسبة", "التسويق", "الاقتصاد"},
    "فن وإعلام": {"فنون", "التصميم", "وسائط متعددة تفاعلية", "العروض"},
    "غير مصنف": {"غير مصنف"},
}

UNIVERSITY_WEIGHTS = {
    "جامعة الطائف": 0.60,
    "جامعة طيبة": 2.09,
    "جامعة جدة": 1.11,
    "جامعة الملك فهد للبترول والمعادن": 0.95,
}


@dataclass(frozen=True)
class QualityOptions:
    missing_fraction_threshold: float = 0.01
    flag_duplicates: bool = True
    flag_straightlining: bool = True
    flag_inconsistent_ai_nonuse: bool = True


def _map_preserving_numeric(series: pd.Series, mapping: dict[str, int | float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = series.astype("string").str.strip().map(mapping)
    return numeric.where(numeric.notna(), mapped)


def group_major(value: object) -> str:
    if pd.isna(value):
        return "تخصص آخر"
    text = str(value).strip()
    for grouped, source_values in MAJOR_GROUPS.items():
        if text in source_values:
            return grouped
    return "تخصص آخر"


def group_university(value: object) -> str:
    if pd.isna(value):
        return "جامعة أخرى"
    text = str(value).strip()
    return text if text in TOP_UNIVERSITIES else "جامعة أخرى"


def assign_university_weight(value: object) -> float:
    if pd.isna(value):
        return 1.0
    text = str(value).strip()
    if text in UNIVERSITY_WEIGHTS:
        return UNIVERSITY_WEIGHTS[text]
    if "الطائف" in text:
        return UNIVERSITY_WEIGHTS["جامعة الطائف"]
    if "طيبة" in text:
        return UNIVERSITY_WEIGHTS["جامعة طيبة"]
    if "جدة" in text:
        return UNIVERSITY_WEIGHTS["جامعة جدة"]
    if "البترول" in text or "المعادن" in text or "KFUPM" in text.upper():
        return UNIVERSITY_WEIGHTS["جامعة الملك فهد للبترول والمعادن"]
    return 1.0


def map_survey_responses(frame: pd.DataFrame, schema: SurveySchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    ensure_columns(frame.columns, schema.required_core_columns(), context="raw survey")
    out = frame.copy()
    out[schema.ai_use] = _map_preserving_numeric(out[schema.ai_use], YES_NO_MAP)
    out[schema.frequency] = _map_preserving_numeric(out[schema.frequency], FREQUENCY_MAP)
    out[schema.contribution] = _map_preserving_numeric(out[schema.contribution], CONTRIBUTION_MAP)

    for column in TASK_SUPPORT_ITEMS + SKILLS_SUPPORT_ITEMS:
        if column in out.columns:
            out[column] = _map_preserving_numeric(out[column], HELP_MAP)
    for column in SKILLS_IMPACT_ITEMS:
        if column in out.columns:
            out[column] = _map_preserving_numeric(out[column], IMPACT_MAP)

    for outcome in (schema.mastery, schema.gpa, schema.confidence):
        out[outcome] = pd.to_numeric(out[outcome], errors="coerce")

    out["الجامعة_المجمعة"] = out[schema.university].map(group_university)
    out["التخصص_المجموع"] = out[schema.major].map(group_major)
    out["analysis_weight"] = out["الجامعة_المجمعة"].map(assign_university_weight).astype(float)
    return out


def _all_equal_nonmissing(row: pd.Series) -> bool:
    values = pd.to_numeric(row, errors="coerce").dropna()
    return len(values) >= 2 and values.nunique() == 1 and float(values.iloc[0]) != 1.0


def build_quality_report(
    frame: pd.DataFrame,
    schema: SurveySchema = DEFAULT_SCHEMA,
    options: QualityOptions = QualityOptions(),
) -> pd.DataFrame:
    report = pd.DataFrame(index=frame.index)
    report["row_index"] = frame.index
    report["duplicate_row"] = frame.duplicated(keep=False) if options.flag_duplicates else False
    report["missing_fraction"] = frame.isna().mean(axis=1)
    report["high_missingness"] = report["missing_fraction"] > options.missing_fraction_threshold

    scale_sets = {
        "task_straightline": [c for c in TASK_SUPPORT_ITEMS if c in frame.columns],
        "skills_support_straightline": [c for c in SKILLS_SUPPORT_ITEMS if c in frame.columns],
        "skills_impact_straightline": [c for c in SKILLS_IMPACT_ITEMS if c in frame.columns],
    }
    for name, columns in scale_sets.items():
        report[name] = frame[columns].apply(_all_equal_nonmissing, axis=1) if options.flag_straightlining and len(columns) >= 2 else False

    if options.flag_inconsistent_ai_nonuse:
        ai_use = pd.to_numeric(frame[schema.ai_use], errors="coerce")
        freq = pd.to_numeric(frame[schema.frequency], errors="coerce")
        contrib = pd.to_numeric(frame[schema.contribution], errors="coerce")
        report["inconsistent_ai_nonuse"] = (ai_use == 0) & ((freq > 1) | (contrib > 1))
    else:
        report["inconsistent_ai_nonuse"] = False

    flag_columns = [c for c in report.columns if c not in {"row_index", "missing_fraction"}]
    report["any_quality_flag"] = report[flag_columns].any(axis=1)
    return report.reset_index(drop=True)


def prepare_survey(
    frame: pd.DataFrame,
    schema: SurveySchema = DEFAULT_SCHEMA,
    *,
    drop_flagged: bool = False,
    explicit_drop_indices: Iterable[int] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = map_survey_responses(frame, schema)
    quality = build_quality_report(mapped, schema)
    drop_set = set(int(i) for i in explicit_drop_indices)
    if drop_flagged:
        drop_set.update(quality.loc[quality["any_quality_flag"], "row_index"].astype(int).tolist())
    cleaned = mapped.drop(index=[i for i in drop_set if i in mapped.index]).reset_index(drop=True)
    return cleaned, quality
