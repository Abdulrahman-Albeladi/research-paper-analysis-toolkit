# Research Toolkit

A recovered research-code portfolio for workflows developed as part of a research-paper assignment sequence. The repository preserves the available analysis logic while separating reusable pipeline code, synthetic demonstration notebooks, and cleaned source notebooks with documented lineage.

## Scope

The repository contains workflows for:

- extracting code samples from repositories;
- preparing and trimming text and Arabic PDF-derived text;
- generating prompt- and sample-oriented analysis inputs;
- running AI-code-detection-oriented analyses;
- examining detector conflicts and high-impact cases;
- conducting textual-pattern and multivariable analyses.

The repository does not include an original research corpus, source PDFs, extracted repository archives, detector outputs, credentials, or other private inputs. The included synthetic notebooks are the supported starting point for reviewing notebook structure without private data.

## Repository layout

```text
src/pipeline/
  code_detection_pipeline.py       Reusable code-detection pipeline material
  repo_code_extraction.py          Repository code-sample extraction material

notebooks/
  analysis_overview_synthetic.ipynb
  multivariable_analysis_synthetic.ipynb
  testing_sample_detection_synthetic.ipynb
  workflows/assignments-workflow.ipynb
  source-cleaned/                  Recovered and cleaned source notebooks

docs/
  data-boundaries.md               Private-data and publication boundaries
  notebook-lineage.md              Mapping and provenance for notebooks
```

## Setup

This repository contains Python source files and Jupyter notebooks, but the supplied file inventory does not establish a pinned dependency set, supported Python version, or reproducible environment specification.

Create an isolated environment and install the packages required by the specific workflow after reviewing its imports:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Then inspect the selected Python module or notebook for its imports before installing dependencies. Use Jupyter only when notebook execution is needed:

```bash
python -m pip install jupyter
jupyter notebook
```

Do not add private corpora, exports, credentials, or API keys to the repository. Use local paths, ignored configuration files, or environment variables where a workflow requires external inputs.

## Starting points

- `notebooks/analysis_overview_synthetic.ipynb` provides a synthetic overview of the analysis workflow.
- `notebooks/multivariable_analysis_synthetic.ipynb` provides a synthetic version of the multivariable analysis workflow.
- `notebooks/testing_sample_detection_synthetic.ipynb` provides a synthetic detection-testing workflow.
- `src/pipeline/code_detection_pipeline.py` and `src/pipeline/repo_code_extraction.py` contain the recovered pipeline-oriented Python code.
- `docs/data-boundaries.md` and `docs/notebook-lineage.md` should be read before using source-cleaned notebooks or attempting to reproduce analyses with non-synthetic inputs.

## Data and privacy boundaries

The analysis-oriented notebooks appear to depend on data that are not distributed here, including material referenced by names such as primary samples, testing samples, assignment corpora, code samples, and Arabic PDFs. Those names describe workflow inputs, not included datasets.

Before running a notebook against non-synthetic data:

1. confirm that the underlying documents, repositories, and derived text may be processed and retained;
2. remove or protect personally identifiable, confidential, licensed, or otherwise restricted material;
3. record local data provenance and preprocessing choices outside version control when they cannot be published;
4. verify any third-party detector, model, or API terms independently; and
5. avoid treating detector output as ground truth without a documented evaluation protocol.

Synthetic notebooks are useful for understanding workflow shape, but they do not validate conclusions on the original private corpus.

## Limitations

- The file inventory does not provide package metadata or a lockfile.
- Original datasets and intermediate artifacts are absent.
- Some source-cleaned notebook names retain duplicated recovery-oriented filenames.
- Notebook outputs and execution assumptions may depend on unavailable local paths, services, packages, or historical data schemas.
- The recovered material supports code and workflow review; it does not, by itself, establish the validity or reproducibility of prior research findings.
- AI-detection workflows require careful interpretation because outputs may vary by detector version, input preparation, and evaluation design.

## Provenance

The repository was assembled from recovered material identified as the "Research Paper (Assignments Part)" collection. Recovered notebooks were retained under `notebooks/source-cleaned/` after cleaning for publication boundaries. Synthetic notebooks provide public-facing examples where private or unavailable inputs would otherwise be required. The notebook naming and lineage record are preserved in `docs/notebook-lineage.md`.

The `notebooks/workflows/assignments-workflow.ipynb` file is retained as a workflow artifact and should not be interpreted as an assignment prompt, grading rubric, or hidden-test suite.

## Contributing

See `CONTRIBUTING.md` for contribution guidance. Before opening a contribution, remove generated notebook noise, caches, local paths, credentials, private inputs, and unreviewed outputs.

## Security and license review

See `SECURITY.md` for vulnerability reporting guidance and `LICENSE_REVIEW.md` for the repository's license-review status.
