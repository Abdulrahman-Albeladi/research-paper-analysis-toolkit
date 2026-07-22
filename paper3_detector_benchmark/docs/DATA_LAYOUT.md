# Data layout and repository fields

## Recommended public data repository

Keep the code repository and data repository separate. The data repository should contain:

```text
data/
  benchmark_manifest.csv
  detector_inputs/
    arabic/{ai_free,ai_generated,humanized_ai}/
    english/{ai_free,ai_generated,humanized_ai}/
    coding/{ai_free,ai_generated,humanized_ai}/
  detector_outputs/
    benchmark_results.csv
  provenance/
    source_manifest.csv
    prompt_workflow.md
    humanization_workflow.md
```

The code repository can reproduce all numerical analyses from `benchmark_results.csv`. Raw API JSON responses may be archived privately if service terms restrict redistribution; the standardized numeric score table is sufficient for the published analyses.

## Result-table schema

Minimum required columns:

```text
Language
Label
file_name
gptzero_ai_rate_percent
pangram_ai_rate_percent
sapling_ai_rate_percent
isgen_ai_rate_percent
```

`word_count` is required for Table 1. Coding rows require only Pangram; the other detector columns may be blank.

## Labels

- `AI-Free`: presumed human-authored material predating November 2022
- `AI-Generated`: directly generated matched sample
- `Humanized AI`: post-generation humanized sample

## Provenance and rights

For every AI-Free source, record the source URL, date, institution/population basis, and redistribution status. Do not upload full public source materials when their licenses do not permit redistribution.
