"""Train a participant-separated, recording-level BITS-2 fall baseline.

Usage:
 python train_odyssey_baseline.py --zip full_dataset.zip --out odyssey_fall_baseline.joblib

This is a laptop model, NOT firmware; one prediction requires a complete recorded activity.
The exported model cannot yet be used as a live streaming fall detector.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, recall_score

SENSORS = ('acg', 'gyro')
STATS = ('mean', 'std', 'p05', 'median', 'p95', 'max_abs', 'mean_abs_step')

def feature_names():
    return [f'{sensor}_{axis}_{stat}' for sensor in SENSORS for axis in ('x', 'y', 'z', 'mag') for stat in STATS]

def extract_features(rows_by_sensor):
    features = []
    for sensor in SENSORS:
        samples = np.asarray(rows_by_sensor[sensor], dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 4 or samples.shape[1] != 3:
            return None
        values = np.column_stack((samples, np.linalg.norm(samples, axis=1)))
        for column in values.T:
            features.extend((np.mean(column), np.std(column), np.percentile(column, 5),
                             np.median(column), np.percentile(column, 95),
                             np.max(np.abs(column)), np.mean(np.abs(np.diff(column)))))
    arr = np.asarray(features, dtype=np.float32)
    return arr if np.isfinite(arr).all() else None

def load_data(zip_path):
    X, y, groups, paths = [], [], [], []
    skip = defaultdict(int)
    with zipfile.ZipFile(zip_path) as archive:
        for path in sorted(archive.namelist()):
            match = re.fullmatch(r'Dataset/(fall|adl)/(user\d+)/[^/]+\.csv', path)
            if not match:
                continue
            activity, user = match.groups()
            samples = defaultdict(list)
            with archive.open(path) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding='utf-8-sig'))
                next(reader, None)
                for row in reader:
                    if len(row) < 6:
                        continue
                    kind = row[5].strip().lower()
                    if kind not in SENSORS:
                        continue
                    try:
                        vec = [float(v) for v in row[1:4]]
                        if np.isfinite(vec).all():
                            samples[kind].append(vec)
                    except ValueError:
                        continue
            v = extract_features(samples)
            if v is None:
                skip[activity] += 1
                continue
            X.append(v)
            y.append(1 if activity == 'fall' else 0)
            groups.append(int(user.removeprefix('user')))
            paths.append(path)
    return np.asarray(X), np.asarray(y), np.asarray(groups), paths, dict(skip)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='full_dataset.zip')
    parser.add_argument('--out', default='odyssey_fall_baseline.joblib')
    parser.add_argument('--report', default='odyssey_fall_results.json')
    args = parser.parse_args()
    X, y, groups, paths, skipped = load_data(args.zip)
    # Deterministic participant split: test users never appear in training or validation.
    train = groups <= 29
    valid = (groups >= 30) & (groups <= 35)
    test = groups >= 36
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    model.fit(X[train], y[train])
    # Keep threshold fixed at 0.5 for this first baseline; validation reserved for future tuning.
    result = {'feature_count': X.shape[1], 'records': len(y), 'skipped': skipped,
              'caution': 'Recording-level classification only; not a streaming or clinically validated fall detector.',
              'splits': {}}
    for name, mask in [('train', train), ('validation', valid), ('test', test)]:
        prediction = model.predict(X[mask])
        tn, fp, fn, tp = confusion_matrix(y[mask], prediction, labels=[0, 1]).ravel()
        result['splits'][name] = {
            'records': int(mask.sum()), 'participants': sorted(set(groups[mask].tolist())),
            'normal_recordings': int(np.sum(y[mask] == 0)), 'fall_recordings': int(np.sum(y[mask] == 1)),
            'accuracy': round(float(accuracy_score(y[mask], prediction)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y[mask], prediction)), 4),
            'fall_recall': round(float(recall_score(y[mask], prediction, pos_label=1)), 4),
            'confusion_matrix': {'true_normal_pred_normal': int(tn), 'true_normal_pred_fall': int(fp),
                                 'true_fall_pred_normal': int(fn), 'true_fall_pred_fall': int(tp)}}
    joblib.dump({'model': model, 'features': feature_names(), 'sensors': SENSORS,
                 'units_note': 'Raw dataset units preserved. Match IMU units before inference.',
                 'scope': 'recording-level offline baseline; not on-device or streaming'}, args.out)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
