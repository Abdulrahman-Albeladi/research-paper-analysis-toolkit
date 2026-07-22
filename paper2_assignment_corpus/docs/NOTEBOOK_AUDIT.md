# Paper 2 source-notebook audit

| Source notebook | Retained components | Removed or reassigned components |
|---|---|---|
| `AI Trimming.ipynb` | generic PDF/DOCX/TXT extraction, paragraph-aware processing, non-authorial block removal, GitHub code-candidate selection | repeated site-specific copies, hard-coded Drive paths, duplicate candidate-picking implementations, manual folder-copy utilities |
| `Trimming Arabic PDF into TXT.ipynb` | Arabic character normalization, paragraph repair, corruption metrics, optional OCR and word-split repair | iLovePDF restart experiments, repeated OCR implementations, fixed folder trees, hard-coded API credentials, one-off recovery branches |
| `Primary_Sample_AI_Detection.ipynb` | detector clients, response normalization, resume-safe collection, result schema, code/Pangram branch | hard-coded keys, duplicated main/testing-sample runs, Google Sheets integration, printed-log reconstruction, path discovery, Pangram credit-specific reruns |
| `AI Code Detection.ipynb` | code chunking and word-weighted Pangram aggregation | repeated full/retry copies, fixed credit calculations, hard-coded keys |
| `AI_Detection_Analysis.ipynb` | weighted score logic, descriptive grouping, Kruskal-Wallis, FDR post hoc comparisons, word-count correlations, broad disagreement summaries | obsolete universal detector weights, raw-PDF copy utility, duplicated reporting/plotting blocks |
| `Paper_2_Adjusted_Multivariable_Analysis_year_fix.ipynb` | year correction, assignment-context construction, overlap/Cramer's V diagnostics, VIF, OLS-HC3, fractional logit, text-only sensitivity models | Drive path candidates, display-only cells, DOCX narrative exporter |
| `Paper_2_High_Impact_Detector_Conflict_Analysis.ipynb` | exact high-impact rule, score range/variance, pairwise patterns, matched comparison logic | duplicated OpenAI branch superseded by the final fixed textual-pattern notebook, fixed paths and figures |
| `Paper_2_OpenAI_Textual_Pattern_Analysis_ASSIGNMENT_CORPUS_FIXED_v3.ipynb` | de-identification, excerpt construction, matched high/non-high conflict comparison, strict JSON-schema feature coding, FDR-corrected prevalence tests | path candidates, fixed output folders, display-only manuscript prose |
| `Generating AI Prompts.ipynb` | not retained in Paper 2 | belongs to the controlled benchmark workflow and is represented in `paper3_detector_benchmark` |
| `Generating AI & Humanized Testing Samples.ipynb` | not retained in Paper 2 | belongs to Paper 3; embedded account credentials removed |

The final Paper 2 code uses language-specific calibration weights from the final manuscript. Earlier detector-level universal weights were intentionally superseded.
