from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .schema import canonical_label, canonical_language, validate_required_columns


def read_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported table format: {suffix}")


def read_text(path: str | Path) -> str:
    path = Path(path)
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_workbook(tables: Mapping[str, pd.DataFrame], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def load_manifest(path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    manifest = read_table(path)
    validate_required_columns(manifest, ["language", "label", "file_name", "file_path"])
    manifest = manifest.copy()
    manifest["language"] = manifest["language"].map(canonical_language)
    manifest["label"] = manifest["label"].map(canonical_label)
    if "file_id" not in manifest.columns:
        manifest["file_id"] = [f"FILE_{i:04d}" for i in range(1, len(manifest) + 1)]

    root_path = Path(root).resolve() if root is not None else None

    def resolve(value: object) -> str:
        path_value = Path(str(value))
        if not path_value.is_absolute() and root_path is not None:
            path_value = root_path / path_value
        return str(path_value.resolve())

    manifest["file_path"] = manifest["file_path"].map(resolve)
    return manifest


def build_manifest_from_config(config_path: str | Path) -> pd.DataFrame:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["benchmark_root"])
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()
    extensions = {ext.lower() for ext in config.get("extensions", [".txt"])}
    rows: list[dict[str, object]] = []
    counters: dict[tuple[str, str], int] = {}

    for language, label_map in config["folder_map"].items():
        language = canonical_language(language)
        for label, relative_folder in label_map.items():
            label = canonical_label(label)
            folder = root / relative_folder
            if not folder.exists():
                raise FileNotFoundError(f"Configured folder does not exist: {folder}")
            key = (language, label)
            counters[key] = 0
            for file_path in sorted(p for p in folder.rglob("*") if p.is_file()):
                if file_path.suffix.lower() not in extensions:
                    continue
                counters[key] += 1
                prefix = {"Arabic": "AR", "English": "EN", "Coding": "CO"}[language]
                class_code = {"AI-Free": "H", "AI-Generated": "A", "Humanized AI": "U"}[label]
                rows.append(
                    {
                        "file_id": f"{prefix}_{class_code}_{counters[key]:03d}",
                        "language": language,
                        "label": label,
                        "file_name": file_path.name,
                        "file_path": str(file_path.relative_to(root)),
                        "source_url": "",
                        "source_date": "",
                        "redistribution_status": "review_required",
                    }
                )
    return pd.DataFrame(rows)
