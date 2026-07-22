from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

from .generation import clean_model_output
from .io_utils import read_text


def chunk_by_paragraphs(text: str, max_words: int = 1500) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start : start + max_words]))
            continue
        if current and current_words + len(words) > max_words:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(paragraph)
        current_words += len(words)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def humanize_text(text: str, *, max_words: int = 1500, retries: int = 3) -> str:
    email = os.environ.get("AI_TEXT_HUMANIZER_EMAIL", "").strip()
    password = os.environ.get("AI_TEXT_HUMANIZER_PASSWORD", "").strip()
    url = os.environ.get("AI_TEXT_HUMANIZER_URL", "https://ai-text-humanizer.com/api.php").strip()
    if not email or not password:
        raise RuntimeError("AI_TEXT_HUMANIZER_EMAIL and AI_TEXT_HUMANIZER_PASSWORD must be set.")
    outputs: list[str] = []
    for chunk_index, chunk in enumerate(chunk_by_paragraphs(text, max_words=max_words), start=1):
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    url,
                    data={"email": email, "pw": password, "text": chunk},
                    timeout=(20, 300),
                )
                response.raise_for_status()
                result = response.text.strip()
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        for key in ("output", "text", "humanized_text", "result", "message"):
                            if isinstance(payload.get(key), str) and payload[key].strip():
                                result = payload[key].strip()
                                break
                except ValueError:
                    pass
                result = clean_model_output(result)
                if not result:
                    raise RuntimeError("Humanizer returned an empty response.")
                outputs.append(result)
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(10 * (2 ** (attempt - 1)), 40))
        else:
            raise RuntimeError(f"Humanizer failed on chunk {chunk_index}: {last_error}")
        time.sleep(2.0)
    return "\n\n".join(outputs)


def process_directory(input_dir: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> None:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    for source in sorted(input_dir.rglob("*.txt")):
        target = output_dir / source.relative_to(input_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            continue
        target.write_text(humanize_text(read_text(source)) + "\n", encoding="utf-8")
