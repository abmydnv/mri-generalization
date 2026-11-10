# MRI Generalization Study

Work in progress: a reproducible study of participant-level evaluation in brain MRI classification, extending a published literature review.

## Current milestone

A manifest validator, participant-level splitter, volume preprocessing, CNN classification pipeline, and an exploratory real-data IXI age-regression pilot. CUDA training has been verified on an RTX 5060. The age pilot uses 561 eligible participants; the CNN checks use synthetic images only.

## First real-data results

| Evaluation | Test participants | Ridge MAE (years) | Training-mean baseline MAE (years) |
| --- | ---: | ---: | ---: |
| Participant-disjoint, site-stratified | 113 | 15.71 | 15.48 |
| Unseen scanner site (IOP) | 68 | 20.89 | 16.29 |

**The initial native-grid ridge model did not beat the simple baseline.** These are exploratory negative results, not clinical performance claims. See [the pilot report](results/ixi-pilot-report.md) for uncertainty, failure analysis, and next steps. Aggregate machine-readable reports are included in `results/`; original MRI, metadata, participant partitions, and models are excluded from Git.

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

Create a Python 3.12 virtual environment and install `requirements.txt`. A CPU-only PyTorch installation is sufficient for tests and smoke training. The local RTX 5060 has been verified with PyTorch 2.7.1+cu128: CUDA forward/backward execution and synthetic CNN training both passed. Exact tested CPU and GPU dependencies are recorded in `requirements-lock.txt` and `requirements-gpu-lock.txt`. Follow https://pytorch.org/get-started/locally/ for other hardware.

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

The active real-data pilot uses openly licensed IXI healthy-volunteer MRI to predict chronological age. It does not classify Alzheimer's disease. See the [IXI protocol](docs/ixi-protocol.md) for predefined evaluation, scanner-site testing, data attribution, and limitations. The binary MRI classification code remains available as a separate implementation foundation.

- Published review: https://doi.org/10.1145/3726122.3726160
- Leakage research: https://pmc.ncbi.nlm.nih.gov/articles/PMC8604922/
- Data access candidate: https://sites.wustl.edu/oasisbrains/request-access/

## Contributions

Abdumannob Muydinov supplies the research context and will study the implementation, run and interpret experiments, and document decisions. Initial implementation and documentation are substantially AI-assisted. Future reports must describe actual contributions and results accurately.
