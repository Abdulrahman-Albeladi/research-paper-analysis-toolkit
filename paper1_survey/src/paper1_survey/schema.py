from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SurveySchema:
    university: str = "حدّد الجامعة التي تنتمي لها"
    gender: str = "يرجى تحديد الجنس"
    major: str = "يرجى تحديد التّخصص\\مجال الدراسة"
    ai_use: str = "هل تستخدم أدوات الذكاء الاصطناعي في دراستك؟"
    frequency: str = "لأي درجة تستخدم الذكاء الاصطناعي لأغراض تعليمية؟"
    contribution: str = "إلى أي مدى يساهم الذكاء الاصطناعي في دراستك الجامعية وإكمال الأعمال والواجبات ؟"
    mastery: str = "كيف تقيّم تأثير استخدام أدوات الذكاء الاصطناعي على أدائك الأكاديمي بإتقان المادة؟ بغض النظر عن المعدّل"
    gpa: str = "كيف تقيّم تأثير استخدام أدوات الذكاء الاصطناعي على معدّلك؟"
    confidence: str = "كيف تؤثر أدوات الذكاء الاصطناعي على ثقتك في أدائك الأكاديمي؟"
    teacher_ai_preference: str = "هل تفضّل طلب المساعدة من معلمك أو استخدام الذكاء الاصطناعي"
    teacher_ai_reason: str = "يرجى شرح سبب تفضيلك للمعلّم أو الذكاء الاصطناعي"
    policy_preference: str = "هل تعتقد أنه يجب السماح للطلاب باستخدام الذكاء الاصطناعي في إنجاز الواجبات والأعمال الدراسية؟"
    policy_reason: str = "يرجى شرح المنطق خلف إجابتك للسؤال السابق"

    @classmethod
    def from_json(cls, path: str | Path) -> "SurveySchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(payload) - set(asdict(cls()).keys())
        if unknown:
            raise ValueError(f"Unknown schema fields: {sorted(unknown)}")
        return cls(**payload)

    def required_core_columns(self) -> list[str]:
        return [
            self.university,
            self.gender,
            self.major,
            self.ai_use,
            self.frequency,
            self.contribution,
            self.mastery,
            self.gpa,
            self.confidence,
        ]


DEFAULT_SCHEMA = SurveySchema()

TASK_SUPPORT_ITEMS = [
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [إتمام الواجبات]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [فهم المواضيع]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [إنجاز الأعمال الكتابية]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [العثور على المصادر]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [الأفكار\\العصف الذهني]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [كتابة المسودة الأولى]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [تلخيص الدروس]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [اختبار فهمي وإتقاني للمواضيع]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في إنجاز كل من الأعمال التالية؟  [العروض\\Presentation]",
]

SKILLS_SUPPORT_ITEMS = [
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في المهارات التالية دراسيًّا؟ [حل المشكلات والمسائل]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في المهارات التالية دراسيًّا؟ [تنظيم وتنسيق الأوراق]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في المهارات التالية دراسيًّا؟ [التفكير النقدي]",
    "لأي درجة تساعدك أدوات الذكاء الاصطناعي في المهارات التالية دراسيًّا؟ [قراءة النصوص الطويلة]",
]

SKILLS_IMPACT_ITEMS = [
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [العصف الذهني]",
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [حل المشكلات والمسائل]",
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [بحث المصدر\\المعلومة]",
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [تنظيم وتنسيق الأوراق]",
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [التفكير النقدي]",
    "لأي درجة يؤثّر استخدامك لأدوات الذكاء الاصطناعي على قدراتك نحو هذه المهارات؟ [قراءة النصوص الطويلة]",
]


def construct_definitions(schema: SurveySchema = DEFAULT_SCHEMA) -> dict[str, list[str]]:
    return {
        "AI_Use_Intensity": [schema.frequency, schema.contribution],
        "AI_Perceived_Benefit": [schema.mastery, schema.gpa, schema.confidence],
        "AI_Task_Support": TASK_SUPPORT_ITEMS.copy(),
        "AI_Skills_Support": SKILLS_SUPPORT_ITEMS.copy(),
        "AI_Skills_Impact": SKILLS_IMPACT_ITEMS.copy(),
    }


def ensure_columns(columns: Iterable[str], required: Iterable[str], *, context: str = "dataset") -> None:
    available = set(columns)
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")
