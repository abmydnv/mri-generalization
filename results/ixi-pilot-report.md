# IXI age prediction: first real-data pilot

## Question and cohort

Can a simple reproducible native-grid MRI intensity model outperform constant mean training age, and does it generalize to an unseen scanner site?

561 eligible adult participants were matched between T1 volumes and valid age metadata. One volume per participant is used. The MRI archive passed MD5 `34901a0593b41dd19c1a1f746eac2d58`. Dataset: [IXI](https://brain-development.org/ixi-dataset/), EPSRC GR/S21533/02, CC BY-SA 3.0. MRI age prediction is not Alzheimer's detection or clinical biological-age estimation.

The preprocessing and evaluation choices are described in [the protocol](../docs/ixi-protocol.md). Standardization and ridge fitting use training data only; five regularization values are selected on validation MAE. No model modification was made in response to test results. Two predefined evaluations use overlapping source participants and must not be treated as independent replications.

## Results

| Evaluation | Train / validation / test | Model MAE | 95% bootstrap MAE interval | Mean-age baseline MAE | Model RMSE | Model R-squared |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Participant-disjoint, site-stratified | 336 / 112 / 113 | 15.71 years | 10.25–24.72 years | 15.48 years | 47.43 years | -6.36 |
| IOP scanner held out | 369 / 124 / 68 | 20.89 years | 17.54–25.48 years | 16.29 years | 26.87 years | -1.66 |

Paired MAE reduction relative to the baseline (positive is better):

- Participant-disjoint: **-0.24 years**, 95% interval **-9.34 to 5.49**. No improvement is established.
- Unseen site: **-4.60 years**, 95% interval **-8.59 to -1.80**. Under this pilot protocol the model performs worse than the baseline.

Intervals use 1,000 participant-bootstrap replicates. They do not include uncertainty from cohort selection, preprocessing choices, or validation-based model selection. These are descriptive exploratory comparisons, not confirmatory statistical tests. Age distributions differ between sites.

## Interpretation and limitations

The first model fails the baseline comparison. Its very large participant-split RMSE relative to MAE indicates substantial errors for some cases; it must not be presented as a successful predictor. The current representation rescales canonical native-space volumes without anatomical registration or skull stripping. Image geometry, non-brain tissue, and acquisition differences are plausible failure mechanisms, but this experiment does not isolate their causal contribution.

Ridge predictions are unbounded. Values were not clipped after examining test outcomes. Healthy volunteers in three London hospitals cannot establish performance in disease cohorts or other populations. The unseen-site experiment also changes age distribution, so its performance gap cannot be attributed solely to scanner hardware.

## Next experimental phase

Preserve these reports, partitions, and failed baseline. Diagnose image quality, geometry, and outliers using training/validation data. Compare preprocessing and robust representations through development-only cross-validation. Record assistance and research decisions. Since these test outcomes have now been observed, future iteration on this cohort is exploratory; a later confirmatory claim requires a genuinely untouched evaluation source or protocol.

## Reproduction and artifacts

```powershell
python download_ixi_parallel.py
python ixi_age.py --output artifacts/ixi-age-pilot-001
python ixi_age.py --holdout-site IOP --output artifacts/ixi-age-site-001
```

Machine-readable aggregate outputs: [participant split](ixi-age-pilot-001.json), [unseen site](ixi-age-site-001.json). Local model parameters, participant partitions, features, and source data are intentionally excluded from the repository. No clinical deployment is proposed.
