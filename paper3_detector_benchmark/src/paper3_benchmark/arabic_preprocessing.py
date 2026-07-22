from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path

from .generation import clean_model_output

ARABIC_TRIM_SYSTEM = """
You are cleaning extracted Arabic academic text for AI-detection preprocessing.

You must not rewrite, paraphrase, summarize, translate, or add words. You may only delete text and repair broken line breaks
by joining lines that clearly belong to the same paragraph.

Keep complete author-written Arabic prose paragraphs, coherent lists, and genuine section titles. Delete cover metadata,
identifiers, repeated headers or footers, tables, captions, references, bibliography, appendices, page numbers, form fields,
template instructions, equation-heavy fragments, corrupted OCR fragments, repeated garbage, and isolated non-substantive text.
Preserve the exact wording and order of retained text. Output only the cleaned text.
""".strip()

OCR_SYSTEM = """
Transcribe the Arabic text visible on this page accurately. Preserve reading order and paragraph boundaries. Include English
text only when it is part of the document. Do not summarize, translate, correct content, or add commentary. Output only the
transcribed page text.
""".strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arabic_ratio(text: str) -> float:
    characters = [char for char in text if not char.isspace()]
    if not characters:
        return 0.0
    arabic = sum(
        ("\u0600" <= char <= "\u06ff")
        or ("\u0750" <= char <= "\u077f")
        or ("\u08a0" <= char <= "\u08ff")
        for char in characters
    )
    return arabic / len(characters)


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 100:
        return False
    if text.endswith((":", "؛", "：")):
        return True
    words = text.split()
    return 1 <= len(words) <= 12 and not re.search(r"[.!?؟]$", text) and _arabic_ratio(text) >= 0.45


def _looks_like_bullet(line: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:[-*•●▪◦‣–—]\s+|\(?[0-9٠-٩]{1,3}\)?[.\-:]\s+|\(?[A-Za-zأ-ي]\)?[.\-:]\s+)",
            line,
        )
    )


def merge_soft_linebreaks(text: str) -> str:
    lines = [line.strip() for line in normalize_text(text).splitlines()]
    output: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            output.append(" ".join(buffer).strip())
            buffer = []

    for line in lines:
        if not line:
            flush()
            continue
        if _looks_like_heading(line) or _looks_like_bullet(line):
            flush()
            output.append(line)
            continue
        if buffer and re.search(r"[.!?؟:؛]$", buffer[-1]):
            flush()
        buffer.append(line)
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def extract_pdf_text(pdf_path: str | Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('Install OCR dependencies with: pip install -e ".[ocr]"') from exc
    document = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in document]
    document.close()
    return merge_soft_linebreaks("\n\n".join(pages))


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install OCR dependencies with: pip install -e ".[ocr]"') from exc
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def ocr_pdf_with_openai(pdf_path: str | Path, *, model: str = "gpt-4.1-mini", dpi: int = 170) -> str:
    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Install OCR dependencies with: pip install -e ".[ocr]"') from exc
    client = _openai_client()
    document = fitz.open(str(pdf_path))
    outputs: list[str] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for page_index in range(len(document)):
        pixmap = document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("jpeg"))).convert("RGB")
        image.thumbnail((1800, 1800))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": OCR_SYSTEM},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        outputs.append(getattr(response, "output_text", "").strip())
        time.sleep(0.4)
    document.close()
    return merge_soft_linebreaks("\n\n".join(outputs))


def trim_with_openai(text: str, *, model: str = "gpt-4.1-mini", chunk_chars: int = 6000) -> str:
    client = _openai_client()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = paragraph if not current else current + "\n\n" + paragraph
    if current:
        chunks.append(current)

    cleaned: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": ARABIC_TRIM_SYSTEM},
                {"role": "user", "content": f"نظّف الجزء رقم {index} حسب التعليمات وأعد فقط النص المحتفَظ به:\n\n{chunk}"},
            ],
        )
        output = clean_model_output(getattr(response, "output_text", ""))
        if output:
            cleaned.append(output)
        time.sleep(1.0)
    return merge_soft_linebreaks("\n\n".join(cleaned))


def process_pdf_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    use_ocr: bool = False,
    use_openai_trim: bool = True,
    overwrite: bool = False,
) -> None:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    for source in sorted(input_dir.rglob("*.pdf")):
        target = output_dir / source.relative_to(input_dir).with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            continue
        extracted = ocr_pdf_with_openai(source) if use_ocr else extract_pdf_text(source)
        cleaned = trim_with_openai(extracted) if use_openai_trim else extracted
        target.write_text(cleaned.strip() + "\n", encoding="utf-8")
