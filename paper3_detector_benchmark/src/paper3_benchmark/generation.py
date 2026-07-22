from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Literal

from .io_utils import read_text

ARABIC_REVERSE_PROMPT_SYSTEM = """
You are helping with a research workflow.

Read a trimmed Arabic academic text and infer one realistic Arabic prompt that could produce a similar finished assignment.
The prompt must sound like a real student request, ask for the deliverable itself, capture the likely topic, structure, tone,
course level, and requirements, and explicitly request approximately the supplied target length. Do not quote long passages.
Do not mention AI detection, reverse engineering, evaluation, or research. Output only the prompt.
""".strip()

ARABIC_GENERATION_SYSTEM = """
Write only the final Arabic assignment requested by the user. Do not add conversational wrappers, disclaimers, notes,
markdown fences, or statements about AI. Output only the assignment text in Arabic.
""".strip()

CODE_REVERSE_PROMPT_SYSTEM = """
Infer one realistic student-style request that could produce code similar to the supplied sample. The request must ask for
building the code itself, capture the project, programming language, features, structure, and difficulty, and request a similar
scope and approximate length. Do not mention AI detection, reverse engineering, evaluation, or research. Output only the prompt.
""".strip()

CODE_GENERATION_SYSTEM = """
Return only the requested code/content. Do not add explanations, notes, conversational wrappers, or markdown fences.
""".strip()

CODE_HUMANIZATION_SYSTEM = """
Rewrite the supplied source code so it appears naturally human-written while preserving functionality and approximately
preserving length. Preserve the programming language and keep the result runnable as far as possible. Do not explain the
changes and do not use markdown fences. Output only the rewritten code.
""".strip()


def clean_model_output(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```[A-Za-z0-9_+-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    wrappers = (
        r"^here(?:'s| is).*?\n+",
        r"^certainly.*?\n+",
        r"^sure.*?\n+",
        r"^إليك.*?\n+",
        r"^بالطبع.*?\n+",
    )
    for pattern in wrappers:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I | re.S).strip()
    return cleaned


class OpenAISampleGenerator:
    def __init__(self, model: str | None = None, sleep_seconds: float = 1.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install optional dependencies with: pip install -e ".[generation]"') from exc
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.sleep_seconds = sleep_seconds

    def _call(self, system: str, user: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        time.sleep(self.sleep_seconds)
        return clean_model_output(getattr(response, "output_text", ""))

    def reverse_prompt(self, text: str, modality: Literal["arabic", "code"], file_name: str) -> str:
        target_length = len(text.split())
        if modality == "arabic":
            system = ARABIC_REVERSE_PROMPT_SYSTEM
            sample = text[:45_000]
            user = f"اسم الملف: {file_name}\nالطول المستهدف التقريبي: {target_length} كلمة\n\nالنص:\n{sample}"
        elif modality == "code":
            system = CODE_REVERSE_PROMPT_SYSTEM
            sample = text[:60_000]
            user = f"File name: {file_name}\nApproximate target length: {target_length} code tokens/words\n\nCode sample:\n{sample}"
        else:
            raise ValueError("Supported modalities are 'arabic' and 'code'.")
        return self._call(system, user)

    def generate(self, prompt: str, modality: Literal["arabic", "code"]) -> str:
        system = ARABIC_GENERATION_SYSTEM if modality == "arabic" else CODE_GENERATION_SYSTEM
        return self._call(system, prompt)

    def humanize_code(self, code: str) -> str:
        return self._call(CODE_HUMANIZATION_SYSTEM, code)


def process_directory(
    input_dir: str | Path,
    prompts_dir: str | Path,
    generated_dir: str | Path,
    *,
    modality: Literal["arabic", "code"],
    humanized_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    input_dir = Path(input_dir)
    prompts_dir = Path(prompts_dir)
    generated_dir = Path(generated_dir)
    humanized_path = Path(humanized_dir) if humanized_dir else None
    generator = OpenAISampleGenerator()

    for source in sorted(input_dir.rglob("*.txt")):
        relative = source.relative_to(input_dir)
        prompt_file = prompts_dir / relative
        generated_file = generated_dir / relative
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        generated_file.parent.mkdir(parents=True, exist_ok=True)
        text = read_text(source)

        if overwrite or not prompt_file.exists():
            prompt = generator.reverse_prompt(text, modality, source.name)
            prompt_file.write_text(prompt + "\n", encoding="utf-8")
        else:
            prompt = read_text(prompt_file)

        if overwrite or not generated_file.exists():
            generated = generator.generate(prompt, modality)
            generated_file.write_text(generated + "\n", encoding="utf-8")
        else:
            generated = read_text(generated_file)

        if modality == "code" and humanized_path is not None:
            target = humanized_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not target.exists():
                target.write_text(generator.humanize_code(generated) + "\n", encoding="utf-8")
