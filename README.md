# MRI Generalization Study

Work in progress: a reproducible study of participant-level evaluation in brain MRI classification, extending a published literature review.

## Current milestone

A manifest validator, deterministic participant-level splitter, local volume-to-slice preprocessing, small CNN training pipeline, majority baseline, participant-level evaluation and bootstrap uncertainty estimates. No real MRI data have been acquired or used for training. Synthetic examples are software checks, not scientific results.

## Run

Requires Python 3.10 or newer. From this directory:

```powershell
python -m unittest discover -s tests -v
python split_manifest.py examples/synthetic.csv --output artifacts/synthetic-splits.csv
```

Input CSV columns: `participant_id,image_path,label`. Labels must be the documented participant-level target, identical for every image of that participant. This first version supports stable participant labels only, not longitudinal label changes. Use local pseudonymous identifiers; never commit real manifests or data.

The splitter stratifies participants by label into approximately 60/20/20 train/validation/test partitions. Small strata may yield different proportions; each label needs at least five participants. Validate cohort composition before research use. Keep the resulting test set fixed and untouched during model selection. Learned preprocessing must use training data only.

## Planned work

1. Verify OASIS access terms, cohort criteria, target labels, and confounders.
2. Establish simple baselines and a GPU-appropriate vision model.
3. Evaluate participant-level predictions with uncertainty and error analysis.
4. Reproduce the known impact of slice-level leakage as an explicitly invalid comparison.

No claims of clinical usefulness or novel leakage discovery are made.

## Training environment

Create a Python 3.12 virtual environment and install `requirements.txt`. A CPU-only PyTorch installation is sufficient for tests and smoke training. Install a compatible CUDA build separately for NVIDIA training, following https://pytorch.org/get-started/locally/. The local CUDA download stalled; GPU training is not yet verified. Exact tested CPU dependency versions are recorded in `requirements-lock.txt` after verification.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe smoke_train.py
```

The smoke test creates random fictitious images under ignored `artifacts/`. Its scores have no scientific meaning. Experiment directories refuse overwrites; choose a new output directory for each run.

For curated real volumes (after data access and cohort review):

```powershell
python prepare_volumes.py data/curated-volumes.csv --root data --output artifacts/prepared
python train.py artifacts/prepared/manifest.csv --root artifacts/prepared --output artifacts/experiment-001
```

The preprocessing assumes documented, aligned volumes; it does not perform skull stripping, registration, or quality control. Labels and cohort selection require a separate dataset-specific audit. Central slices are a baseline representation. The CNN averages slice probabilities within each participant, selects its checkpoint on validation balanced accuracy, and evaluates the held-out test partition once. The default 200 bootstrap replicates are an initial diagnostic; small cohorts produce unstable intervals. GPU reproducibility is not guaranteed across hardware or library versions.

Data access status and a draft research statement: [data-access.md](docs/data-access.md).

## Background

- Published review: https://doi.org/10.1145/3726122.3726160
- Leakage research: https://pmc.ncbi.nlm.nih.gov/articles/PMC8604922/
- Data access candidate: https://sites.wustl.edu/oasisbrains/request-access/

## Contributions

Abdumannob Muydinov supplies the research context and will study the implementation, run and interpret experiments, and document decisions. Initial implementation and documentation are substantially AI-assisted. Future reports must describe actual contributions and results accurately.
