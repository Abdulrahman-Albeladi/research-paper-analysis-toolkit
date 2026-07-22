# Paper 1 data layout

Recommended local layout:

```text
data/
├── raw/
│   ├── survey.csv
│   ├── pilot_old.xlsx
│   └── pilot_edited.xlsx
└── processed/
    └── survey_mapped.csv
```

The `raw/` and `processed/` directories are ignored by Git. Do not publish participant identifiers or identifiable free-text responses.
