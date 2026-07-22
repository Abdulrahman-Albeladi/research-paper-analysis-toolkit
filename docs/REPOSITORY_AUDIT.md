# Cross-paper notebook-to-repository audit

This monorepository was reconstructed from the supplied Colab notebooks and the previously cleaned Paper 3 repository.

## Consolidation principles

- retain code that directly supports reported methods, tables, diagnostics, or reproducibility;
- convert notebook cells into reusable functions and command-line scripts;
- remove repeated versions after retaining the latest complete implementation;
- replace Google Drive path-candidate searches with explicit CLI arguments;
- remove package-install cells, display-only cells, and one-off repair runs;
- remove embedded secrets and require environment variables;
- exclude exploratory branches that did not contribute to a manuscript result;
- document proprietary-service and data-sharing limitations.

Detailed source mappings are provided inside each paper directory.
