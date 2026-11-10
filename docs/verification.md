# Verification: implementation milestone 2

## Subsequent GPU and age-pilot checks

The RTX 5060 now passes CUDA matrix multiplication and backpropagation with PyTorch 2.7.1+cu128. A two-epoch synthetic CNN training run completed on CUDA. Twelve unit tests now pass, adding age metric and fixed-grid feature tests. No broken package requirements were reported. The original CPU results below remain historical verification records.

The openly licensed IXI T1 archive was subsequently downloaded and its published checksum verified. Both predefined age-regression experiments completed on 561 matched eligible participants. Their aggregate outputs and negative results are published in results/. This is real-data ridge regression; the CNN has still only been smoke-tested on synthetic data.

Local platform: Windows, Python 3.12, CPU PyTorch 2.15.0+cpu. NVIDIA RTX 5060 detected by nvidia-smi; CUDA training has not been verified. The CUDA wheel download stalled and was cancelled; CPU installation completed successfully.

- Ten unit tests passed: participant-disjoint splits, deterministic assignments, malformed input rejection, training manifest leakage checks, duplicate paths, missing classes, known-value metrics, bootstrap intervals, volume slice extraction, and constant-volume rejection.
- A two-epoch end-to-end training run completed on random fictitious images, including checkpoint selection, checkpoint reload, participant-level aggregation, majority baseline, and uncertainty calculation.
- Local smoke artifacts and generated images are ignored by Git.
- No real MRI data acquired, no real disease classification experiment run, and no clinical performance established.
- CI workflow added; remote CI status must be verified separately.

Commands:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe smoke_train.py
```

The smoke command requires that `artifacts/smoke-run` does not already exist. Existing experiment directories are deliberately protected against overwrites.
