# Source-notebook audit

The repository was reconstructed from five supplied notebooks.

| Source notebook | Retained components | Removed or consolidated components |
|---|---|---|
| `Testing_Sample_AI_Detection(1).ipynb` | API clients, response normalization, resume-safe collection, binary/soft metrics | hard-coded keys, repeated Pangram reruns, Google Drive setup, Google Sheets-only output, obsolete path assumptions |
| `Paper_3(1).ipynb` | Cochran's Q and exact McNemar tests with BH adjustment | fixed workbook path and one-off execution block |
| `conflict_analysis_testing_sample(1).ipynb` | canonical conflict fields, vote-pattern classification, score dispersion, feature extraction, Mann-Whitney tests, optional qualitative review | Drive/workbook discovery, duplicate output exports, plotting-only cells, workbook/ZIP consolidation utility |
| `Extracting_Code_Samples(1).ipynb` | GitHub URL parsing, clone-and-filter workflow, code-file scoring, candidate selection | hard-coded repository lists, extra-repository search, PDF link extraction, Colab mounting, duplicated implementations |
| `Arabic&Coding_AI_Pipeline(1).ipynb` | Arabic cleaning/reverse prompting, matched generation, code reverse prompting/generation, code humanization, paragraph chunking | embedded credentials, multiple OCR experiments, one-off missing-file patches, word-count reports, HTML-review utilities, duplicate API experiments, fixed Drive paths |

The final methodological-weight implementation follows the formula reported in the final manuscript: AI-Free and AI-generated condition scores each receive exponent 0.4, while humanized-AI receives exponent 0.2, followed by within-language normalization.
