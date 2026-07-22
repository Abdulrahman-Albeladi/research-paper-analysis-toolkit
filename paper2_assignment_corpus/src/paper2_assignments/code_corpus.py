from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

CODE_SUFFIXES = {".py", ".java", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".jsx", ".tsx", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".scala", ".r", ".sql"}
EXCLUDED_PARTS = {"node_modules", ".git", "vendor", "dist", "build", "target", "__pycache__", ".venv", "venv"}
GENERATED_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock", "composer.lock"}


def parse_github_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError(f"Not a GitHub URL: {url}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"GitHub URL does not identify a repository: {url}")
    return parts[0], parts[1].removesuffix(".git")


def clone_repository(url: str, destination: str | Path) -> Path:
    destination = Path(destination)
    subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True, capture_output=True, text=True)
    return destination


def is_candidate_code_file(path: Path) -> bool:
    if path.suffix.lower() not in CODE_SUFFIXES:
        return False
    if any(part.lower() in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.name.lower() in GENERATED_NAMES or path.stat().st_size > 1_000_000:
        return False
    return True


def score_code_file(path: Path, text: str) -> float:
    score = min(len(text.split()) / 200, 20)
    score += 5 if re.search(r"\b(class|def|function|public|private|interface|struct)\b", text) else 0
    score += 5 if re.search(r"\b(if|for|while|switch|try|catch|return)\b", text) else 0
    score += 4 if re.search(r"\b(import|include|using|require|from)\b", text) else 0
    score -= 10 if len(set(text.splitlines())) < max(3, len(text.splitlines()) // 5) else 0
    score += 3 if path.name.lower() in {"main.py", "main.java", "app.py", "app.js", "index.js"} else 0
    return float(score)


def collect_repository_code(repository: str | Path, max_files: int = 30, max_words: int = 6000) -> list[tuple[str, str]]:
    repository = Path(repository)
    candidates: list[tuple[float, Path, str]] = []
    for path in repository.rglob("*"):
        if not path.is_file() or not is_candidate_code_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        candidates.append((score_code_file(path, text), path, text))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str]] = []
    total_words = 0
    for _, path, text in candidates:
        words = len(text.split())
        if selected and total_words + words > max_words:
            continue
        selected.append((path.relative_to(repository).as_posix(), text))
        total_words += words
        if len(selected) >= max_files or total_words >= max_words:
            break
    return selected


def format_multifile_submission(files: list[tuple[str, str]]) -> str:
    blocks = [f"===== FILE: {relative_path} =====\n{text.strip()}" for relative_path, text in files]
    return "\n\n".join(blocks).strip()


def prepare_github_submission(url: str, output_path: str | Path, max_files: int = 30, max_words: int = 6000) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        repository = clone_repository(url, Path(temporary) / "repository")
        files = collect_repository_code(repository, max_files=max_files, max_words=max_words)
        if not files:
            raise RuntimeError(f"No suitable code files found in {url}")
        output_path.write_text(format_multifile_submission(files), encoding="utf-8")
    return output_path


def split_code_by_words(text: str, max_words: int = 2500) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for line in lines:
        line_words = len(line.split())
        if current and words + line_words > max_words:
            chunks.append("\n".join(current))
            current, words = [], 0
        current.append(line)
        words += line_words
    if current:
        chunks.append("\n".join(current))
    return chunks


def aggregate_chunk_scores(scores: list[float], word_counts: list[int]) -> float:
    if not scores or len(scores) != len(word_counts):
        raise ValueError("Scores and word counts must be non-empty and equal length.")
    total = sum(word_counts)
    if total <= 0:
        return float(sum(scores) / len(scores))
    return float(sum(score * count for score, count in zip(scores, word_counts)) / total)
