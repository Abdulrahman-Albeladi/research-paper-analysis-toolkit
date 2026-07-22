from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .detector_clients import DetectorClients
from .io_utils import load_manifest, read_text, write_csv
from .schema import tools_for_language

OUTPUT_COLUMNS = [
    "file_id",
    "Language",
    "Label",
    "file_name",
    "word_count",
    "gptzero_ai_rate_percent",
    "gptzero_result",
    "pangram_ai_rate_percent",
    "pangram_result",
    "sapling_ai_rate_percent",
    "sapling_result",
    "isgen_ai_rate_percent",
    "isgen_result",
    "collection_status",
    "collection_error",
]


def count_words(text: str) -> int:
    return len(text.split())


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\x00", " ").splitlines()).strip()


def truncate_at_boundary(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    paragraph = candidate.rfind("\n\n")
    if paragraph > int(max_chars * 0.6):
        return candidate[:paragraph].strip()
    line = candidate.rfind("\n")
    if line > int(max_chars * 0.6):
        return candidate[:line].strip()
    return candidate.strip()


def _existing_output(path: Path) -> pd.DataFrame:
    if path.exists():
        current = pd.read_csv(path, encoding="utf-8-sig")
        for column in OUTPUT_COLUMNS:
            if column not in current.columns:
                current[column] = np.nan if column.endswith("_percent") or column == "word_count" else ""
        return current[OUTPUT_COLUMNS]
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def collect_detector_scores(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    files_root: str | Path | None = None,
    max_text_chars: int = 50_000,
    sleep_between_calls: float = 2.0,
    resume: bool = True,
) -> pd.DataFrame:
    manifest = load_manifest(manifest_path, root=files_root)
    output_path = Path(output_path)
    output = _existing_output(output_path)
    clients = DetectorClients()

    for _, item in manifest.iterrows():
        file_id = str(item["file_id"])
        existing_index = output.index[output["file_id"].astype(str).eq(file_id)].tolist()
        if existing_index and resume and output.loc[existing_index[0], "collection_status"] == "complete":
            continue

        index = existing_index[0] if existing_index else len(output)
        text = clean_text(read_text(item["file_path"]))
        language = item["language"]
        detector_text = text if language == "Coding" else truncate_at_boundary(text, max_text_chars)
        row = {column: "" for column in OUTPUT_COLUMNS}
        row.update(
            {
                "file_id": file_id,
                "Language": language,
                "Label": item["label"],
                "file_name": item["file_name"],
                "word_count": count_words(text),
                "collection_status": "incomplete",
                "collection_error": "",
            }
        )
        errors: list[str] = []
        for tool in tools_for_language(language):
            try:
                if tool == "isgen":
                    result = clients.isgen(detector_text, language)
                else:
                    result = clients.get(tool)(detector_text)
                row[f"{tool}_ai_rate_percent"] = result.score_percent
                row[f"{tool}_result"] = result.label
            except Exception as exc:
                errors.append(f"{tool}: {type(exc).__name__}: {exc}")
            time.sleep(sleep_between_calls)

        row["collection_status"] = "complete" if not errors else "partial"
        row["collection_error"] = " | ".join(errors)
        for column in OUTPUT_COLUMNS:
            output.loc[index, column] = row.get(column, "")
        write_csv(output[OUTPUT_COLUMNS], output_path)

    return output[OUTPUT_COLUMNS]
