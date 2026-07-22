# Paper 1 source-notebook audit

| Source notebook | Retained components | Removed or consolidated components |
|---|---|---|
| `Pilot_Test.ipynb` | reliability, item diagnostics, KMO/Bartlett, one-factor EFA, old-versus-edited scale comparison | Drive mounting, package-install cells, fixed path search, display-only narrative cells |
| `Adjusted_Survey_Data_Analysis.ipynb` | Arabic response mappings, university/major grouping, quality flags, post-stratification weights, descriptive summaries, construct scores, reliability/EFA, ordered-logit models, PO diagnostics, ablation analysis, open-response theme coding, validation workbook | repeated plotting implementations, superseded t-test/ANOVA branches, multiple copies of ordinal helpers, path-candidate searches, embedded OpenAI key, large notebook outputs, one-off manual row-removal blocks |

The final public code does not silently drop quality-flagged records. Row removal must be requested explicitly through the preparation script so the decision remains auditable.
