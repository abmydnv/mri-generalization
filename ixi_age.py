"""Exploratory IXI brain-age baseline. Healthy volunteers; not disease detection."""
import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import xlrd
from scipy.ndimage import zoom
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def score(y, prediction):
    return dict(mae_years=float(mean_absolute_error(y, prediction)),
                rmse_years=float(np.sqrt(mean_squared_error(y, prediction))),
                r2=float(r2_score(y, prediction)))


def intervals(y, prediction, seed=42):
    rng = np.random.default_rng(seed)
    values = [mean_absolute_error(y[i], prediction[i])
              for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    return np.percentile(values, [2.5, 97.5]).tolist()


def paired_gain(y, prediction, baseline, seed=42):
    """Positive gain means lower absolute error than the training-mean baseline."""
    difference = np.abs(y - baseline) - np.abs(y - prediction)
    rng = np.random.default_rng(seed)
    means = [float(difference[rng.integers(0, len(y), len(y))].mean()) for _ in range(1000)]
    return dict(mae_reduction_years=float(difference.mean()),
                paired_bootstrap_95=np.percentile(means, [2.5, 97.5]).tolist())


def features(path):
    volume = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Expected finite 3D MRI')
    foreground = volume[volume > 0]
    if not len(foreground):
        raise ValueError('Empty foreground')
    low, high = np.percentile(foreground, [1, 99])
    if high <= low:
        raise ValueError('Constant volume')
    scaled = np.clip((volume - low) / (high - low), 0, 1)
    # Native canonical grid resized to a fixed grid: exploratory, not registered anatomy.
    return zoom(scaled, np.array([16, 16, 16]) / np.array(volume.shape), order=1).ravel()


def metadata(path):
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = sheet.row_values(0)
    records = {}
    conflicts = set()
    for index in range(1, sheet.nrows):
        row = dict(zip(header, sheet.row_values(index)))
        try:
            pid, age = int(row['IXI_ID']), float(row['AGE'])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(age) or age < 18 or age > 100:
            continue
        if pid in records and records[pid] != age:
            conflicts.add(pid)
        records[pid] = age
    return {pid: age for pid, age in records.items() if pid not in conflicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ixi'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout-site', choices=['Guys', 'HH', 'IOP'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    volumes = args.data / 'volumes'
    if volumes.exists() and not (volumes / '.complete').exists():
        raise ValueError('Incomplete extraction directory; inspect before retrying')
    if not volumes.exists():
        archive = args.data / 'IXI-T1.tar'
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'md5').hexdigest()
        if digest != '34901a0593b41dd19c1a1f746eac2d58':
            raise ValueError('IXI archive checksum mismatch')
        volumes.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(volumes, filter='data')
        (volumes / '.complete').write_text(digest, encoding='ascii')
    ages = metadata(args.data / 'IXI.xls')
    records, seen = [], set()
    for path in sorted(volumes.rglob('*.nii.gz')):
        match = re.fullmatch(r'IXI(\d+)-(Guys|HH|IOP)-\d+-T1.nii.gz', path.name)
        if not match:
            continue
        pid, site = int(match[1]), match[2]
        if pid not in ages:
            continue
        if pid in seen:
            raise ValueError('Duplicate participant volume')
        seen.add(pid)
        records.append(dict(participant_id=f'IXI{pid:03d}', site=site, age=ages[pid], path=path))
    if len(records) < 50:
        raise ValueError('Too few matched participants for pilot')
    indices = np.arange(len(records))
    if args.holdout_site:
        test = indices[[r['site'] == args.holdout_site for r in records]]
        development = indices[[r['site'] != args.holdout_site for r in records]]
        train, validation = train_test_split(development, test_size=.25, random_state=args.seed)
    else:
        development, test = train_test_split(indices, test_size=.2, random_state=args.seed,
                                             stratify=[r['site'] for r in records])
        train, validation = train_test_split(development, test_size=.25, random_state=args.seed,
                                             stratify=[records[i]['site'] for i in development])
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + (args.data / 'IXI.xls').read_bytes())
    for record in records:
        stat = record['path'].stat()
        fingerprint.update(f'{record["path"].name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    cache_key = fingerprint.hexdigest()
    cache = args.data / f'features-{cache_key}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            X = saved['features']
        if X.shape != (len(records), 4096) or not np.isfinite(X).all():
            raise ValueError('Invalid feature cache')
    else:
        vectors = []
        for index, record in enumerate(records):
            vectors.append(features(record['path']))
            if (index + 1) % 50 == 0:
                print(f'Processed {index + 1}/{len(records)} volumes', flush=True)
        X = np.stack(vectors)
        np.savez_compressed(cache, features=X)
    y = np.array([r['age'] for r in records])
    best, selected, history = float('inf'), None, []
    for alpha in (1., 10., 100., 1000., 10000.):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[train], y[train])
        mae = float(mean_absolute_error(y[validation], model.predict(X[validation])))
        history.append(dict(alpha=alpha, validation_mae_years=mae))
        if mae < best:
            best, selected, selected_alpha = mae, model, alpha
    prediction = selected.predict(X[test])
    baseline = np.full(len(test), y[train].mean())
    report = dict(dataset='IXI T1', disease_detection=False, seed=args.seed, feature_cache_key=cache_key,
                  holdout_site=args.holdout_site, participants=len(records), selected_alpha=selected_alpha,
                  partitions={name: dict(count=len(group), sites=dict(Counter(records[i]['site'] for i in group)),
                                         age_mean=float(y[group].mean()), age_min=float(y[group].min()),
                                         age_max=float(y[group].max()))
                              for name, group in [('train', train), ('validation', validation), ('test', test)]},
                  validation_search=history, test=score(y[test], prediction),
                  test_mae_bootstrap_95=intervals(y[test], prediction),
                  training_mean_baseline=score(y[test], baseline),
                  comparison_to_mean_baseline=paired_gain(y[test], prediction, baseline),
                  limitations=['Exploratory native-grid intensity baseline; no registration or skull stripping.',
                               'Single dataset, healthy volunteers only; no clinical or Alzheimer validation.',
                               'Participant-disjoint results do not establish robustness to other populations.'],
                  source='https://brain-development.org/ixi-dataset/', license='CC BY-SA 3.0')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(args.output / 'model.npz', coef=selected[-1].coef_, intercept=selected[-1].intercept_,
                        scaler_mean=selected[0].mean_, scaler_scale=selected[0].scale_)
    with (args.output / 'partition.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['participant_id', 'split'])
        for name, group in [('train', train), ('validation', validation), ('test', test)]:
            writer.writerows((records[i]['participant_id'], name) for i in group)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
