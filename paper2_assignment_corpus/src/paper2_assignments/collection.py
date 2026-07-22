from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .code_corpus import aggregate_chunk_scores, split_code_by_words
from .detector_clients import DetectorClients
from .io_utils import read_text, write_csv
from .scoring import canonical_language

TOOLS_BY_LANGUAGE = {
    "Arabic": ("gptzero", "pangram", "sapling", "isgen"),
    "English": ("gptzero", "pangram", "sapling", "isgen"),
    "Code": ("pangram",),
}

OUTPUT_COLUMNS = [
    "file_id", "file_name", "file_path", "language", "university", "major", "year_original",
    "assignment_type", "word_count",
    "gptzero_ai_rate_percent", "gptzero_result",
    "pangram_ai_rate_percent", "pangram_result",
    "sapling_ai_rate_percent", "sapling_result",
    "isgen_ai_rate_percent", "isgen_result",
    "collection_status", "collection_error",
]


def count_words(text: str) -> int:
    return len(text.split())


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def truncate_at_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    for token in ("\n\n", "\n", ". "):
        boundary = candidate.rfind(token)
        if boundary > int(max_chars * 0.6):
            return candidate[: boundary + (1 if token == ". " else 0)].strip()
    return candidate.strip()


def _existing_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    current = pd.read_csv(path, encoding="utf-8-sig")
    for column in OUTPUT_COLUMNS:
        if column not in current.columns:
            current[column] = np.nan if column.endswith("_percent") or column == "word_count" else ""
    return current[OUTPUT_COLUMNS]


def _run_tool(clients: DetectorClients, tool: str, text: str, language: str):
    if tool == "isgen":
        return clients.isgen(text, language)
    return clients.get(tool)(text)


def _collect_code_pangram(clients: DetectorClients, text: str, max_words_per_chunk: int = 2500):
    chunks = split_code_by_words(text, max_words=max_words_per_chunk)
    scores: list[float] = []
    word_counts: list[int] = []
    labels: list[str] = []
    for chunk in chunks:
        result = clients.pangram(chunk)
        scores.append(result.score_percent)
        labels.append(result.label)
        word_counts.append(max(1, count_words(chunk)))
    score = aggregate_chunk_scores(scores, word_counts)
    label = "Likely AI" if score >= 50 else "Likely Human"
    return score, label


def collect_detector_scores(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    files_root: str | Path | None = None,
    max_text_chars: int = 50_000,
    sleep_between_calls: float = 2.0,
    resume: bool = True,
) -> pd.DataFrame:
    files_root = Path(files_root) if files_root else None
    output_path = Path(output_path)
    output = _existing_output(output_path)
    clients = DetectorClients()

    for item in manifest.to_dict(orient="records"):
        file_id = str(item.get("file_id") or item.get("file_name"))
        existing_indices = output.index[output["file_id"].astype(str).eq(file_id)].tolist()
        if existing_indices and resume and output.loc[existing_indices[0], "collection_status"] == "complete":
            continue
        index = existing_indices[0] if existing_indices else len(output)
        path_value = item.get("processed_path") or item.get("file_path")
        file_path = Path(str(path_value))
        if not file_path.is_absolute() and files_root is not None:
            file_path = files_root / file_path
        language = canonical_language(item.get("language", "English"))
        text = clean_text(read_text(file_path))
        detector_text = text if language == "Code" else truncate_at_boundary(text, max_text_chars)

        row = {column: "" for column in OUTPUT_COLUMNS}
        row.update(
            {
                "file_id": file_id,
                "file_name": item.get("file_name", file_path.name),
                "file_path": str(path_value),
                "language": language,
                "university": item.get("university", ""),
                "major": item.get("major", ""),
                "year_original": item.get("year_grouping") or item.get("year_original", ""),
                "assignment_type": item.get("assignment_type", ""),
                "word_count": count_words(text),
                "collection_status": "incomplete",
                "collection_error": "",
            }
        )
        errors: list[str] = []
        for tool in TOOLS_BY_LANGUAGE[language]:
            try:
                if language == "Code" and tool == "pangram":
                    score, label = _collect_code_pangram(clients, detector_text)
                    row["pangram_ai_rate_percent"] = score
                    row["pangram_result"] = label
                else:
                    result = _run_tool(clients, tool, detector_text, language)
                    row[f"{tool}_ai_rate_percent"] = result.score_percent
                    row[f"{tool}_result"] = result.label
            except Exception as error:
                errors.append(f"{tool}: {type(error).__name__}: {error}")
            time.sleep(sleep_between_calls)
        row["collection_status"] = "complete" if not errors else "partial"
        row["collection_error"] = " | ".join(errors)
        for column in OUTPUT_COLUMNS:
            output.loc[index, column] = row.get(column, "")
        write_csv(output[OUTPUT_COLUMNS], output_path)
    return output[OUTPUT_COLUMNS]
