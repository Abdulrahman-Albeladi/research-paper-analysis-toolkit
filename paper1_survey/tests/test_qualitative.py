from paper1_survey.qualitative import normalize_policy_stance, normalize_teacher_ai_stance, redact_pii


def test_stance_and_redaction():
    assert normalize_teacher_ai_stance("أفضل الذكاء الاصطناعي") == "Prefer Artificial Intelligence"
    assert normalize_policy_stance("نعم ولكن بشروط") == "Yes, but with Conditions"
    redacted = redact_pii("email test@example.com and 0551234567")
    assert "test@example.com" not in redacted
    assert "0551234567" not in redacted
