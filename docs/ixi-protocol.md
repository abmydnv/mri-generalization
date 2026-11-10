# IXI pilot protocol

## Scope

The active real-data pilot predicts chronological age from healthy-volunteer T1 MRI. It tests the reproducibility and generalization workflow without requiring institutional data access. It does not detect Alzheimer's disease, estimate biological age, or establish clinical usefulness. The earlier binary-classification pipeline remains a separate implementation example.

## Source and permissions

IXI: https://brain-development.org/ixi-dataset/ (EPSRC GR/S21533/02). Dataset license: CC BY-SA 3.0. Acknowledge IXI in every report. Raw and derived participant-level data remain outside Git. Do not redistribute data adaptations without complying with the dataset's license.

T1 archive distributed in MONAI's public-dataset tutorial: https://developer.download.nvidia.com/assets/Clara/monai/tutorials/IXI-T1.tar

Archive MD5: `34901a0593b41dd19c1a1f746eac2d58`, also recorded by the TorchIO IXI loader. This checksum checks transfer integrity, not scientific quality.

Metadata comes from the official IXI.xls spreadsheet. Use only participant identifier and age for matching; do not publish other spreadsheet fields. Exclude missing/invalid ages, non-adults, conflicting duplicate ages, and unmatched images. Reject duplicate participant volumes.

## Predefined evaluation

- Seed 42. Split participants 60/20/20 into training/validation/test, stratifying by scanner site to preserve site representation.
- Resize canonical-oriented volume intensities to 16 x 16 x 16. Normalize each volume with its own positive-foreground intensity percentiles. This does not use cohort-level information.
- Fit feature standardization only on training participants. Fit ridge regression with alpha in {1,10,100,1000,10000}; select on validation MAE only. No refitting or tuning against test data.
- Compare with constant mean training age. Report test MAE and RMSE in years, R-squared, and 1,000 participant-bootstrap percentile replicates for the MAE interval.
- Report a paired participant-bootstrap interval for MAE reduction relative to the constant baseline. Positive reduction favors the model. Do not claim superiority if the interval crosses zero.
- A predefined secondary experiment holds out IOP as an unseen scanner site. Randomly split the remaining sites into training/validation at 75/25. Do not select the held-out site based on results.

## Limitations

This is an intentionally simple baseline using native image grids. It does not align brain anatomy through registration or remove non-brain tissue, so spatial position, acquisition protocol, head size, and other confounders can influence prediction. Scanner sites and age distributions may differ. Healthy volunteers do not represent disease cohorts. A low error would not establish clinical validity; an error worse than the constant baseline remains a reportable finding.

## Reproduction

```powershell
python download_ixi_parallel.py
python ixi_age.py --output artifacts/ixi-age-pilot-001
python ixi_age.py --holdout-site IOP --output artifacts/ixi-age-site-001
```

Downloads are approximately 4.8 GB. Completed experiment directories refuse overwrites. No participant-level file should be committed to the public repository. Reports intended for publication must contain only aggregate results and limitations.
