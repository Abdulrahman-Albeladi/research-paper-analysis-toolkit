# Security and reproducibility notes

## Credential removal

The source notebooks contained credentials for multiple external services. The cleaned repository reads all credentials from environment variables and never writes them to logs, exported tables, or configuration files.

Before publishing this repository:

1. Revoke or rotate every OpenAI, GPTZero, Pangram, Sapling, RapidAPI/Isgen, GitHub, and external-humanizer credential that appeared in the original notebooks.
2. Confirm that no notebook, Colab export, shell history, or Git commit containing an old credential is pushed to GitHub.
3. Run `python scripts/check_repository_for_secrets.py .`.
4. Review the GitHub repository after pushing, including commit history and Actions logs.

## Proprietary service versions

The commercial detectors and humanizers are black-box services. Their models, thresholds, interfaces, and response schemas can change without notice. Archive the exact detector-output CSV used in the paper. That archived output is the authoritative input for reproduction of the reported statistics.

## Raw-file redistribution

Public accessibility does not automatically grant redistribution rights. For AI-Free source materials:

- retain original source URLs and access dates;
- record repository/article licenses where available;
- redistribute processed content only where permitted;
- otherwise provide a source manifest and the derived numeric detector outputs rather than copyrighted full text/code.

## Deterministic analysis

The quantitative analysis is deterministic given the archived result table. Optional OpenAI-assisted qualitative review is not deterministic and should be treated as interpretive support, not as a source of benchmark labels, accuracy values, conflict categories, or detector weights.
