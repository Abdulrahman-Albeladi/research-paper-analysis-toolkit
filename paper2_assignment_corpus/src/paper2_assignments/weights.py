from __future__ import annotations

import json
from pathlib import Path

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "Arabic": {"gptzero": 0.24, "pangram": 0.33, "sapling": 0.16, "isgen": 0.27},
    "English": {"gptzero": 0.27, "pangram": 0.28, "sapling": 0.24, "isgen": 0.21},
    "Code": {"pangram": 1.0},
}


def load_weights(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    payload = DEFAULT_WEIGHTS if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, dict[str, float]] = {}
    for language, tool_weights in payload.items():
        total = sum(float(value) for value in tool_weights.values())
        if total <= 0:
            raise ValueError(f"Weights for {language} do not sum to a positive value.")
        normalized[language] = {str(tool).lower(): float(value) / total for tool, value in tool_weights.items()}
    return normalized
