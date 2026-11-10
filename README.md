# MRI Generalization Study

Work in progress: a reproducible study of participant-level evaluation in brain MRI classification, extending a published literature review.

## Current milestone

A dependency-free manifest validator and deterministic participant-level splitter. This prevents images from one participant appearing in multiple partitions and rejects inconsistent labels and duplicate image paths. No MRI data have been acquired and no models have been trained. Synthetic examples are software checks, not scientific results.

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

## Background

- Published review: https://doi.org/10.1145/3726122.3726160
- Leakage research: https://pmc.ncbi.nlm.nih.gov/articles/PMC8604922/
- Data access candidate: https://sites.wustl.edu/oasisbrains/request-access/

## Contributions

Abdumannob Muydinov supplies the research context and will study the implementation, run and interpret experiments, and document decisions. Initial implementation and documentation are substantially AI-assisted. Future reports must describe actual contributions and results accurately.
