# Verification: implementation milestone 2

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
