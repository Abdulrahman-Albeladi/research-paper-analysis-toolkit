# Paper 2: Authentic Assignment Corpus Analysis

This directory contains cleaned Python code supporting the manuscript:

> **AI-Writing Signals in Saudi University Assignments: A Detector-Based Analysis**

## Included workflows

- recursive manifest construction for Arabic, English, and code submissions;
- plain-text extraction from TXT, DOCX, and PDF files;
- Arabic encoding cleanup, paragraph repair, and optional OCR/OpenAI-assisted boundary trimming;
- code-submission parsing and Pangram chunk aggregation;
- resume-safe collection of GPTZero, Pangram, Sapling, and Isgen outputs;
- final language-specific calibration weights reported in the manuscript;
- descriptive tables by university, major, year, language, and assignment type;
- Kruskal-Wallis tests and FDR-adjusted pairwise comparisons;
- word-count correlations and detector-specific English-versus-Arabic comparisons;
- robust OLS with HC3 errors and fractional-logit sensitivity models;
- overlap diagnostics, Cramer's V, and VIF checks;
- broad and high-impact detector-conflict definitions;
- optional OpenAI-assisted textual-feature coding on de-identified excerpts.

## Final detector weights

The cleaned code uses the final Paper 3 calibration-informed weights reported in Paper 2:

| Submission type | GPTZero | Pangram | Sapling | Isgen |
|---|---:|---:|---:|---:|
| Arabic text | 0.24 | 0.33 | 0.16 | 0.27 |
| English text | 0.27 | 0.28 | 0.24 | 0.21 |
| Code | — | 1.00 | — | — |

The earlier notebook branch that used detector-level overall accuracies as universal weights is intentionally not retained because it does not match the final manuscript.

## Typical workflow

```bash
pip install -e ".[api,ocr,dev]"

python scripts/01_build_manifest.py \
  --root data/raw/assignments \
  --output data/assignment_manifest.csv

python scripts/02_preprocess_assignments.py \
  --manifest data/assignment_manifest.csv \
  --output-root data/processed/text \
  --output-manifest data/processed_manifest.csv

python scripts/03_collect_detector_scores.py \
  --manifest data/processed_manifest.csv \
  --output data/detector_results.csv

python scripts/04_run_primary_analysis.py \
  --input data/detector_results.csv \
  --output outputs/primary

python scripts/05_run_adjusted_models.py \
  --input outputs/primary/assignment_level_analysis.csv \
  --output outputs/adjusted

python scripts/06_run_conflict_analysis.py \
  --input outputs/primary/assignment_level_analysis.csv \
  --output outputs/conflict
```

## Data and privacy

The authentic student-assignment corpus is not included and should not be uploaded to GitHub. The public repository should contain only code, schemas, manifests without private paths, synthetic examples, and aggregate outputs approved for release.

## Reproducibility boundary

Detector APIs are commercial black boxes and may change. Archive the exact detector-output dataset used in the paper under the journal-approved data-access arrangement; analysis on that archived dataset is deterministic.
