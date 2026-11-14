"""Train an experimental fixed-window model from the extracted BITS-2 CSVs.

The supplied baseline script/model are unavailable in this workspace. This is a
new model, saved separately, and must not be compared to baseline metrics.
"""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from backend.features import FEATURE_NAMES, WINDOW_SIZE, extract_features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'ml' / 'full_dataset' / 'Dataset'


def load_recording(path):
    streams = {'acg': [], 'gyro': []}
    with path.open(newline='') as handle:
        for row in csv.reader(handle):
            if len(row) < 6 or row[5] not in streams:
                continue
            try:
                values = [float(v) for v in row[1:4]]
            except ValueError:
                continue
            if np.all(np.isfinite(values)):
                streams[row[5]].append(values)
    if min(map(len, streams.values())) < WINDOW_SIZE:
        return None
    acc = np.asarray(streams['acg'])  # gravity included, m/s²
    gyro = np.asarray(streams['gyro'])  # rad/s
    positions = np.linspace(0, 1, len(acc))
    source = np.linspace(0, 1, len(gyro))
    aligned_gyro = np.column_stack([np.interp(positions, source, gyro[:, i]) for i in range(3)])
    return np.column_stack((acc, aligned_gyro))


def build_dataset():
    groups = {'train': ([], []), 'validation': ([], []), 'test': ([], [])}
    paths = sorted(DATASET.glob('*/user*/*.csv'))
    users = sorted({p.parent.name for p in paths}, key=lambda s: int(s[4:]))
    if len(users) < 5:
        raise RuntimeError('Too few participants for grouped split')
    rng = np.random.default_rng(26)
    rng.shuffle(users)
    n = len(users)
    split = {u: ('train' if i < int(.6*n) else 'validation' if i < int(.8*n) else 'test')
             for i, u in enumerate(users)}
    counts = {'recordings_skipped': 0, 'participants': {k: sorted([u for u in users if split[u] == k]) for k in groups}}
    for path in paths:
        data = load_recording(path)
        if data is None:
            counts['recordings_skipped'] += 1
            continue
        # Non-overlapping windows avoid near-duplicate samples within a recording.
        for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
            x, y = groups[split[path.parent.name]]
            x.append(extract_features(data[start:start + WINDOW_SIZE]))
            y.append(path.parent.parent.name)
    return groups, counts


def main():
    groups, counts = build_dataset()
    train_x, train_y = groups['train']
    if not train_x or len(set(train_y)) != 2:
        raise RuntimeError('Insufficient labeled training windows')
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                   class_weight='balanced_subsample', random_state=26, n_jobs=-1)
    model.fit(np.asarray(train_x), train_y)
    result = {'method': '32-sample non-overlapping windows; participant-level split',
              'feature_names': FEATURE_NAMES, 'window_size': WINDOW_SIZE, **counts}
    for name, (x, y) in groups.items():
        predicted = model.predict(np.asarray(x))
        result[name] = {'windows': len(y), 'accuracy': accuracy_score(y, predicted),
                        'fall_recall': recall_score(y, predicted, pos_label='fall'),
                        'confusion_matrix_adl_fall': confusion_matrix(y, predicted, labels=['adl', 'fall']).tolist()}
    joblib.dump({'model': model, 'feature_names': FEATURE_NAMES, 'window_size': WINDOW_SIZE},
                ROOT / 'ml' / 'odyssey_window_model.joblib')
    (ROOT / 'ml' / 'odyssey_window_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('train', 'validation', 'test')}, indent=2))


if __name__ == '__main__':
    main()
