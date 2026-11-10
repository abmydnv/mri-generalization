"""Generate fictitious images and exercise the complete training pipeline."""
import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from split_manifest import partition

root = Path('artifacts/smoke-data')
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
rows = []
for label in ('0', '1'):
    for person in range(10):
        for image in range(2):
            name = f'fictional-{label}-{person}-{image}.png'
            array = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
            Image.fromarray(array).save(root / name)
            rows.append(dict(participant_id=f'fictional-{label}-{person}', image_path=name, label=label))
with (root / 'manifest.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['participant_id', 'image_path', 'label', 'split'])
    writer.writeheader()
    writer.writerows(partition(rows))
subprocess.run([sys.executable, 'train.py', str(root / 'manifest.csv'), '--root', str(root),
                '--output', 'artifacts/smoke-run', '--epochs', '2', '--synthetic'], check=True)
