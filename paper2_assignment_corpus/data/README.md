# Paper 2 data layout

Recommended local layout:

```text
data/
├── raw/assignments/             # private; never commit
├── assignment_manifest.csv
├── processed/text/              # de-identified detector inputs; private
├── processed_manifest.csv
└── detector_results.csv         # archive under approved data-access conditions
```

The manifest should contain one row per assignment and the columns shown in `manifest_template.csv`. Use relative paths when creating a shareable manifest.
