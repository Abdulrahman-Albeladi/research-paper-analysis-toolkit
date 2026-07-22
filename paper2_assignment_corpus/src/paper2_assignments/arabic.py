from __future__ import annotations

import base64
import json
import os
import re
import unicodedata
from pathlib import Path

ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u066E-\u06D3]")


def normalize_arabic_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ھ", "ه")
    text = text.replace("\ufeff", "").replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = merge_soft_linebreaks(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 120:
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*|[أ-ي]|[-•])\s*[.)-]?\s+", text):
        return True
    if text.endswith(":") or text.endswith("؟"):
        return True
    words = text.split()
    return len(words) <= 8 and not re.search(r"[.!،؛]$", text)


def merge_soft_linebreaks(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    output: list[str] = []
    buffer = ""
    for line in lines:
        current = line.strip()
        if not current:
            if buffer:
                output.append(buffer.strip())
                buffer = ""
            output.append("")
            continue
        if looks_like_heading(current) or re.match(r"^[-•*]\s+", current):
            if buffer:
                output.append(buffer.strip())
                buffer = ""
            output.append(current)
            continue
        if buffer:
            buffer += " " + current
        else:
            buffer = current
        if re.search(r"[.!؟؛:]$", current):
            output.append(buffer.strip())
            buffer = ""
    if buffer:
        output.append(buffer.strip())
    return "\n".join(output)


def arabic_character_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(ARABIC_LETTER_RE.fullmatch(character)) for character in letters) / len(letters)


def suspicious_token_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 1.0
    suspicious = 0
    for token in tokens:
        if "�" in token or "□" in token or len(token) > 40:
            suspicious += 1
        elif re.search(r"[A-Za-z][\u0600-\u06FF]|[\u0600-\u06FF][A-Za-z]", token):
            suspicious += 1
    return suspicious / len(tokens)


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def ocr_image_with_openai(path: str | Path, *, model: str | None = None) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install the OCR dependency group.") from error
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for optional OCR.")
    client = OpenAI()
    response = client.responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Transcribe all visible Arabic and English text accurately. Preserve paragraphs. Return only the transcription; do not summarize or correct meaning."},
                    {"type": "input_image", "image_url": image_to_data_url(path)},
                ],
            }
        ],
    )
    return normalize_arabic_text(response.output_text)


def repair_split_arabic_words_with_openai(text: str, *, model: str | None = None) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install the API dependency group.") from error
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for optional word-split repair.")
    client = OpenAI()
    prompt = (
        "Repair only Arabic words that were incorrectly split by spaces or OCR corruption. Preserve wording, order, punctuation, and paragraph structure. "
        "Do not paraphrase, summarize, add, or remove substantive content. Return only the repaired text.\n\n" + text
    )
    response = client.responses.create(model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), input=prompt)
    return normalize_arabic_text(response.output_text)
