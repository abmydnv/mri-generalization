"""Convert a curated local binary-label volume manifest to central axial PNGs.

Input: participant_id,image_path,label. Volume paths are relative to --root.
Choose one documented preprocessed volume per participant before using this tool.
"""
import argparse
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from split_manifest import partition


def extract_slices(path):
    image = nib.as_closest_canonical(nib.load(path))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3 or min(volume.shape) < 8 or not np.isfinite(volume).all():
        raise ValueError('Expected a finite 3D volume with at least eight voxels per axis')
    foreground = volume[volume > 0]
    if foreground.size == 0:
        raise ValueError('No positive foreground')
    low, high = np.percentile(foreground, [1, 99])
    if high <= low:
        raise ValueError('Constant foreground intensity')
    # Per-volume normalization has no fitted population parameters.
    volume = np.clip((volume - low) / (high - low), 0, 1)
    indices = np.unique(np.linspace(int(volume.shape[2] * .35), int(volume.shape[2] * .65), 8).astype(int))
    return [(int(index), (volume[:, :, index].T * 255).astype(np.uint8)) for index in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if len({r['participant_id'] for r in rows}) != len(rows):
        raise ValueError('Provide exactly one curated volume per participant')
    # Validate fields and cohort size before writing any images.
    partition(rows)
    if any(r['label'] not in ('0', '1') for r in rows):
        raise ValueError('Expected binary labels 0/1')
    args.output.mkdir(parents=True, exist_ok=False)
    slices = []
    for i, row in enumerate(rows):
        for index, array in extract_slices(args.root / row['image_path']):
            # Filenames avoid leaking source participant identifiers.
            name = f'volume-{i:05d}-slice-{index:03d}.png'
            Image.fromarray(array).save(args.output / name)
            slices.append(dict(participant_id=row['participant_id'], image_path=name, label=row['label']))
    with (args.output / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['participant_id', 'image_path', 'label', 'split'])
        writer.writeheader()
        writer.writerows(partition(slices))
    print(f'Prepared {len(slices)} slices from {len(rows)} participant volumes')


if __name__ == '__main__':
    main()
