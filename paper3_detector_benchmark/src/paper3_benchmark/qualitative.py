from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .io_utils import read_text


def run_qualitative_review(
    analysis_master: pd.DataFrame,
    output_path: str | Path,
    *,
    max_files_per_group: int = 5,
    model: str | None = None,
    max_chars: int = 8000,
) -> pd.DataFrame:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[generation]"') from exc
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    review_groups = [
        ("Arabic Humanized conflict", (analysis_master["Language"] == "Arabic") & (analysis_master["Label"] == "Humanized AI") & analysis_master["is_text_conflict"]),
        ("English Humanized conflict", (analysis_master["Language"] == "English") & (analysis_master["Label"] == "Humanized AI") & analysis_master["is_text_conflict"]),
        ("Arabic AI-Free conflict", (analysis_master["Language"] == "Arabic") & (analysis_master["Label"] == "AI-Free") & analysis_master["is_text_conflict"]),
        ("Coding AI false negatives", (analysis_master["Language"] == "Coding") & (analysis_master["truth_class"] == "AI") & (analysis_master["pangram_pred_label"] == "Human")),
    ]
    rows: list[dict[str, object]] = []
    for group_name, mask in review_groups:
        sample = analysis_master[mask].sort_values("text_score_range", ascending=False, na_position="last").head(max_files_per_group)
        for _, row in sample.iterrows():
            content = read_text(row["file_path"])[:max_chars]
            prompt = f"""You are reviewing a benchmark file from an AI-detection conflict study.

1. Briefly describe the dominant structural and stylistic patterns.
2. Explain why such a file might cause detector conflict or a false negative.
3. Remain factual and concise.
4. Do not speculate beyond the supplied content.

Language: {row['Language']}
Benchmark label: {row['Label']}
File identifier: {row['file_name']}

File content:
{content}
"""
            response = client.responses.create(model=model, input=prompt, temperature=0.2)
            rows.append(
                {
                    "Review_Group": group_name,
                    "Language": row["Language"],
                    "Label": row["Label"],
                    "file_name": row["file_name"],
                    "Model_Review": getattr(response, "output_text", ""),
                }
            )
    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
