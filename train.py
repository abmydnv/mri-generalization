"""Train a small CNN on a validated, participant-disjoint slice manifest."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Slices(Dataset):
    def __init__(self, rows, root):
        self.rows, self.root = rows, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image_path']) as source:
            image = np.asarray(source.convert('L').resize((96, 96)), dtype=np.float32) / 255
        return torch.from_numpy(image.copy())[None], int(row['label']), row['participant_id']


def validate(rows):
    if not rows:
        raise ValueError('Empty manifest')
    participants, paths = {}, set()
    for row in rows:
        if row['label'] not in ('0', '1') or row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Expected binary labels and train/validation/test partitions')
        pid = row['participant_id']
        if not pid.strip():
            raise ValueError('Empty participant ID')
        value = (row['label'], row['split'])
        if pid in participants and participants[pid] != value:
            raise ValueError('Participant leakage or conflicting labels')
        participants[pid] = value
        path = str(Path(row['image_path'])).casefold()
        if path in paths:
            raise ValueError('Duplicate image path')
        paths.add(path)
    for split in ('train', 'validation', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'0', '1'}:
            raise ValueError(f'{split} must contain both classes')


def participant_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Invalid probabilities')
    predictions = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                auroc=float(roc_auc_score(labels, probabilities)), sensitivity=float(tp / (tp + fn)),
                specificity=float(tn / (tn + fp)))


def bootstrap_intervals(labels, probabilities, seed=42, repetitions=200):
    """Resample independent participants; skip undefined single-class replicates."""
    rng = np.random.default_rng(seed)
    samples = []
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            samples.append(participant_metrics(labels[indices], probabilities[indices]))
    if not samples:
        raise ValueError('No valid bootstrap replicates')
    return dict(method='participant percentile bootstrap', requested_replicates=repetitions,
                valid_replicates=len(samples), seed=seed,
                intervals_95={key: np.percentile([s[key] for s in samples], [2.5, 97.5]).tolist()
                              for key in samples[0]})


def evaluate(model, loader, device, uncertainty=False):
    subjects = {}
    model.eval()
    with torch.no_grad():
        for images, labels, ids in loader:
            probabilities = model(images.to(device)).softmax(1)[:, 1].cpu().tolist()
            for pid, label, probability in zip(ids, labels.tolist(), probabilities):
                subjects.setdefault(pid, (label, []))[1].append(probability)
    labels = [v[0] for v in subjects.values()]
    probabilities = [float(np.mean(v[1])) for v in subjects.values()]
    result = dict(participants=len(labels), **participant_metrics(labels, probabilities))
    if uncertainty:
        result['uncertainty'] = bootstrap_intervals(labels, probabilities)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--synthetic', action='store_true', help='Mark outputs as synthetic software checks')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with args.manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    validate(rows)
    # Refuse overwrites of completed experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = {s: DataLoader(Slices([r for r in rows if r['split'] == s], args.root),
                             batch_size=16, shuffle=(s == 'train'), num_workers=0)
               for s in ('train', 'validation', 'test')}
    model = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, history = -1, []
    for epoch in range(args.epochs):
        model.train()
        for images, labels, _ in loaders['train']:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, loaders['validation'], device)
        history.append(dict(epoch=epoch + 1, validation=metrics))
        if metrics['balanced_accuracy'] > best:
            best = metrics['balanced_accuracy']
            torch.save(model.state_dict(), args.output / 'best.pt')
    model.load_state_dict(torch.load(args.output / 'best.pt', weights_only=True, map_location=device))
    train_subjects = {r['participant_id']: int(r['label']) for r in rows if r['split'] == 'train'}
    majority_probability = float(np.mean(list(train_subjects.values())) >= .5)
    test_subjects = {r['participant_id']: int(r['label']) for r in rows if r['split'] == 'test'}
    majority = participant_metrics(list(test_subjects.values()), [majority_probability] * len(test_subjects))
    report = dict(synthetic=args.synthetic, seed=args.seed, device=str(device), torch=torch.__version__,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), history=history,
                  majority_baseline=majority, test=evaluate(model, loaders['test'], device, uncertainty=True),
                  limitations='Small baseline; no clinical validation. Synthetic runs are not research results.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
