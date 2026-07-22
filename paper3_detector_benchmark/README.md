# Multilingual AI-Detector Benchmark: Reproducible Code

This repository contains the cleaned, journal-ready Python code supporting the manuscript:

> **A Multilingual Benchmark of Commercial AI Detection Tools for Arabic, English, and Code Assignments in Saudi Higher Education**

The original research code was distributed across several Google Colab notebooks. This repository removes notebook-only setup, duplicated cells, Google Drive path discovery, hard-coded credentials, unused exploratory fragments, and output-consolidation utilities. The retained code is organized into reusable `.py` modules and command-line scripts.

## What this repository covers

- extraction and deletion-only trimming of Arabic PDF text, with optional image OCR;
- preparation of code samples from public GitHub repositories;
- reverse-prompt generation and matched AI-sample generation for Arabic and code;
- optional English humanization through the external API used in the study;
- code humanization with the OpenAI API;
- collection and normalization of GPTZero, Pangram, Sapling, and Isgen outputs;
- threshold-based and continuous-score performance metrics;
- methodological detector weights based on AI-Free, AI-generated, and humanized-AI conditions;
- Cochran's Q tests and exact McNemar post hoc comparisons with Benjamini-Hochberg adjustment;
- detector-conflict patterns, score dispersion, and structural-feature comparisons;
- optional OpenAI-assisted qualitative review of selected conflict cases.

## What is not included

The raw 450-file benchmark is **not** included in this code archive. Upload the data separately to the journal-approved repository and provide a manifest using `data/manifest_template.csv`. The original notebooks are also intentionally excluded because they contained obsolete paths, repeated code, and embedded credentials.

Arabic humanization with WriteHuman.ai was conducted through the external service workflow described in the manuscript and was not represented by reusable source code in the supplied notebooks. English humanization is supported by `scripts/06_humanize_english_text.py` when valid service credentials are supplied.

## Security notice

The supplied notebooks contained hard-coded service credentials. Those values have been removed completely from this repository. **Rotate/revoke every key or password that appeared in the original notebooks before publishing this repository.** See `docs/SECURITY_AND_REPRODUCIBILITY.md`.

## Repository layout

```text
.
├── config/benchmark.example.json
├── data/manifest_template.csv
├── docs/
├── scripts/
│   ├── 01_build_manifest.py
│   ├── 02_collect_detector_scores.py
│   ├── 03_run_benchmark_analysis.py
│   ├── 04_prepare_code_corpus.py
│   ├── 05_generate_matched_samples.py
│   ├── 06_humanize_english_text.py
│   ├── 07_optional_qualitative_review.py
│   ├── 08_prepare_arabic_corpus.py
│   └── check_repository_for_secrets.py
├── src/paper3_benchmark/
└── tests/
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[generation,dev]"
```

## Data manifest

Create a CSV with one row per benchmark file. Required columns are:

```text
file_id,language,label,file_name,file_path
```

Canonical values:

- `language`: `Arabic`, `English`, or `Coding`
- `label`: `AI-Free`, `AI-Generated`, or `Humanized AI`

`file_path` may be absolute or relative to a root passed on the command line. Optional provenance columns such as `source_url`, `source_date`, and `redistribution_status` are strongly recommended.

## Typical workflow

### 1. Build a manifest from a configured folder layout

```bash
python scripts/01_build_manifest.py \
  --config config/benchmark.example.json \
  --output data/benchmark_manifest.csv
```

### 2. Collect detector scores

Set credentials as environment variables. Never place keys in code or commit a `.env` file.

```bash
export GPTZERO_API_KEY="..."
export PANGRAM_API_KEY="..."
export SAPLING_API_KEY="..."
export ISGEN_RAPIDAPI_KEY="..."

python scripts/02_collect_detector_scores.py \
  --manifest data/benchmark_manifest.csv \
  --output data/benchmark_results.csv
```

Arabic and English use all four detectors. Coding uses Pangram only, matching the study design. The script saves after every file and resumes safely.

### 3. Reproduce the quantitative analysis

```bash
python scripts/03_run_benchmark_analysis.py \
  --results data/benchmark_results.csv \
  --manifest data/benchmark_manifest.csv \
  --output outputs
```

The analysis produces CSV tables and a multi-sheet Excel workbook containing:

- dataset audit and class balance;
- mean post-trimming word counts;
- overall and category-specific detector metrics;
- methodological detector scores and normalized weights;
- Cochran's Q and exact McNemar results;
- detector-conflict summaries and dominant signatures;
- score-dispersion tests;
- optional structural-feature comparisons when file paths are available.

### 4. Run tests and the secret scan

```bash
pytest -q
python scripts/check_repository_for_secrets.py .
```

## Reproducibility boundaries

Commercial detector APIs and humanization services are proprietary black-box systems and may change after the study period. The API clients in this repository reflect the response shapes and endpoints used in the supplied notebooks. Re-running the collection stage at a later date may not reproduce the same detector scores. The analysis stage is deterministic when run on the archived detector-output dataset.

## Citation

A `CITATION.cff` file is provided. Update the article DOI after publication.

## License

No code license is imposed in this package because ownership and institutional requirements should be confirmed by all authors before public release. Add an appropriate open-source license, such as MIT or BSD-3-Clause, before publishing on GitHub if approved by the author group.
