# Security and credential handling

The source notebooks contained embedded credentials for some combination of OpenAI, GPTZero, Pangram, Sapling, Isgen/RapidAPI, iLovePDF, and an external humanization service. No credential values are retained in this repository.

Before making the repository public:

1. Revoke or rotate every credential that appeared in an original notebook.
2. Search the full Git history, not only the current working tree.
3. Keep replacement credentials in environment variables or an untracked `.env` file.
4. Do not commit Google Drive mount paths, account identifiers, API responses containing private URLs, or service dashboards.
5. Run `python scripts/check_repository_for_secrets.py .` before every release.

The scanner is a safeguard, not a guarantee. A manual review is still required.
