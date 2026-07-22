import pandas as pd

from paper1_survey.cleaning import assign_university_weight, map_survey_responses
from paper1_survey.schema import DEFAULT_SCHEMA


def test_mapping_and_groups():
    s = DEFAULT_SCHEMA
    frame = pd.DataFrame(
        {
            s.university: ["جامعة الطائف", "جامعة غير مدرجة"],
            s.gender: ["ذكر", "أنثى"],
            s.major: ["علوم الحاسب", "الطب"],
            s.ai_use: ["نعم", "لا"],
            s.frequency: ["يوميًّا", "لا أستخدمه"],
            s.contribution: ["أعتمد عليه بشكل كبير", "لا أستخدمه"],
            s.mastery: [5, 2],
            s.gpa: [4, 2],
            s.confidence: [5, 3],
        }
    )
    out = map_survey_responses(frame)
    assert out[s.ai_use].tolist() == [1, 0]
    assert out["التخصص_المجموع"].tolist() == ["تقنية وبيانات", "صحة"]
    assert out["الجامعة_المجمعة"].tolist() == ["جامعة الطائف", "جامعة أخرى"]
    assert assign_university_weight("جامعة الطائف") == 0.60
