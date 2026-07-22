# Paper 1: Survey Analysis

This directory contains cleaned Python code supporting the manuscript:

> **Artificial Intelligence Use Among Undergraduates in Selected Saudi Universities: Adoption, Attitudes, and Perceived Academic Effects**

## Included workflows

- mapping Arabic survey responses to numeric values;
- university and major grouping;
- duplicate, missingness, inconsistent-response, and straightlining checks;
- university post-stratification weights used in the source analysis;
- weighted and unweighted descriptive tables;
- Cronbach's alpha, Spearman-Brown reliability, KMO, Bartlett's test, and one-factor EFA summaries;
- mean composite scores for AI Use Intensity, Perceived Benefit, Task Support, Skills Support, and Skills Impact;
- ordered-logit models for mastery, GPA, and confidence outcomes;
- proportional-odds diagnostics and ablation comparisons;
- optional supervised multi-label coding of the two open-response questions;
- optional human-validation workbook generation.

## Data inputs

The anonymized survey CSV is not included. Use `config/survey_columns.example.json` as the canonical schema. The scripts accept explicit paths and do not search Google Drive.

## Typical workflow

```bash
pip install -e ".[qualitative,dev]"

python scripts/01_prepare_survey_data.py \
  --input data/raw/survey.csv \
  --output data/processed/survey_mapped.csv \
  --quality-report outputs/quality_report.csv

python scripts/02_run_pilot_psychometrics.py \
  --old-survey data/raw/pilot_old.xlsx \
  --edited-survey data/raw/pilot_edited.xlsx \
  --output outputs/pilot

python scripts/03_run_main_survey_analysis.py \
  --input data/processed/survey_mapped.csv \
  --output outputs/main

python scripts/04_code_open_responses.py \
  --input data/processed/survey_mapped.csv \
  --output outputs/open_responses
```

Open-response coding requires `OPENAI_API_KEY` and should be run only on de-identified text under the approved ethics/data-handling conditions.

## Reproducibility note

The original notebooks included several evolving versions of the same analyses. This repository retains the final construct definitions, ordered-logit approach, robustness checks, and qualitative coding scheme used to support the manuscript while removing superseded ANOVA/t-test branches and duplicate display/export cells.
