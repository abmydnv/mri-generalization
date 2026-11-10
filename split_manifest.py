"""Validate an MRI manifest and partition whole participants, not images."""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def partition(rows, seed=42):
    """Return copied rows with a split; reject ambiguous participant data."""
    if not rows:
        raise ValueError('Manifest is empty')
    participants = {}
    paths = set()
    for row in rows:
        for field in ('participant_id', 'image_path', 'label'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing or empty field: {field}')
            if row[field] != row[field].strip():
                raise ValueError(f'Leading/trailing whitespace in {field}')
        pid, label = row['participant_id'], row['label']
        path = row['image_path'].replace('\\', '/').casefold()
        if path in paths:
            raise ValueError('Duplicate image path')
        paths.add(path)
        if pid in participants and participants[pid] != label:
            raise ValueError(f'Conflicting labels for participant {pid}')
        participants[pid] = label
    groups = defaultdict(list)
    for pid, label in participants.items():
        groups[label].append(pid)
    assignment = {}
    rng = random.Random(seed)
    for label in sorted(groups):
        ids = sorted(groups[label])
        if len(ids) < 5:
            raise ValueError(f'Label {label} needs at least five participants')
        rng.shuffle(ids)
        n_holdout = max(1, round(len(ids) * 0.2))
        for i, pid in enumerate(ids):
            assignment[pid] = ('test' if i < n_holdout else
                               'validation' if i < 2 * n_holdout else 'train')
    return [dict(row, split=assignment[row['participant_id']]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with args.manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    result = partition(rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing fixed test partition.
    with args.output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['participant_id', 'image_path', 'label', 'split'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result)
    for split in ('train', 'validation', 'test'):
        subset = [r for r in result if r['split'] == split]
        print(f'{split}: {len({r["participant_id"] for r in subset})} participants, {len(subset)} images')


if __name__ == '__main__':
    main()
