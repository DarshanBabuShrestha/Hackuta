"""Trial-level replay and calibration for real wrist gesture CSV recordings.

Rows are grouped by movement_id, replayed at recorded Arduino timestamps, and
never split across calibration and held-out evaluation subsets.
"""
import argparse
import csv
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import random
import sys

from backend.serial_bridge import Sample
from backend.wrist_gesture import WristGestureConfig, WristGestureRecognizer


def load_trials(path, expected_label):
    required = {'timestamp_ms', 'label', 'movement_id', 'instruction',
                'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'}
    grouped = {}
    closed_trials = set()
    previous_trial_id = None
    with Path(path).open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing required CSV fields {sorted(required - set(reader.fieldnames or []))}')
        for row in reader:
            trial_id = row['movement_id'].strip()
            if not trial_id:
                raise ValueError(f'{path}: blank movement_id')
            if previous_trial_id is not None and trial_id != previous_trial_id:
                closed_trials.add(previous_trial_id)
            if trial_id in closed_trials:
                raise ValueError(f'{path}: movement_id {trial_id} appears in separate row blocks')
            previous_trial_id = trial_id
            values = tuple(float(row[key]) for key in
                           ('ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'))
            grouped.setdefault(trial_id, []).append((int(row['timestamp_ms']) % (2 ** 32),
                                                       row['label'].strip(),
                                                       row['instruction'].strip(), values))

    trials = []
    for trial_id, rows in grouped.items():
        labels = {row[1] for row in rows}
        instructions = {row[2] for row in rows}
        if labels != {expected_label}:
            raise ValueError(f'{path}: trial {trial_id} has unexpected/mixed labels: {sorted(labels)}')
        if len(instructions) != 1:
            raise ValueError(f'{path}: trial {trial_id} has inconsistent instructions')
        samples = [Sample(timestamp, values) for timestamp, _, _, values in rows]
        deltas = [((samples[i].timestamp_ms - samples[i - 1].timestamp_ms) % (2 ** 32))
                  for i in range(1, len(samples))]
        duration_ms = sum(deltas)
        if len(samples) < 100 or duration_ms < 2300 or duration_ms > 2700:
            raise ValueError(f'{path}: trial {trial_id} incomplete ({len(samples)} samples, {duration_ms} ms)')
        if not deltas or max(deltas) > 100 or min(deltas) == 0:
            raise ValueError(f'{path}: trial {trial_id} has duplicate or missing timestamp intervals')
        magnitudes = [sum(value * value for value in sample.values[3:]) ** 0.5
                      for sample in samples]
        trailing_stillness_ms = 0
        if magnitudes[-1] <= 18.0:
            trailing = len(samples) - 1
            while trailing > 0 and magnitudes[trailing - 1] <= 18.0:
                trailing -= 1
            cadence_ms = sorted(deltas)[len(deltas) // 2]
            trailing_stillness_ms = sum(deltas[trailing:]) + cadence_ms
        trials.append({'trial_id': trial_id, 'label': expected_label,
                       'instruction': next(iter(instructions)), 'samples': samples,
                       'duration_ms': duration_ms,
                       'peak_gyro_magnitude_dps': max(magnitudes),
                       'trailing_stillness_ms': trailing_stillness_ms})

    if not trials:
        raise ValueError(f'{path}: no trial rows')
    # Ensure movement_id segments are independent captures separated by the
    # prompted pauses/countdown, rather than adjacent labels in one recording.
    seen = set()
    for index, trial in enumerate(trials):
        if trial['trial_id'] in seen:
            raise ValueError(f'{path}: duplicate trial id {trial["trial_id"]}')
        seen.add(trial['trial_id'])
        if index:
            previous = trials[index - 1]['samples'][-1].timestamp_ms
            current = trial['samples'][0].timestamp_ms
            separation_ms = (current - previous) % (2 ** 32)
            if separation_ms < 1000:
                raise ValueError(f'{path}: trial {trial["trial_id"]} is not separated from its predecessor')
    return trials


def replay_trial(trial, config):
    recognizer = WristGestureRecognizer(config)
    start_ms = trial['samples'][0].timestamp_ms
    for sample in trial['samples']:
        if recognizer.feed(sample, confirming=True):
            latency = (sample.timestamp_ms - start_ms) % (2 ** 32)
            return {'detected': True, 'latency_ms': latency}
    return {'detected': False, 'latency_ms': None}


def metric(trials, predictions):
    tp = fp = tn = fn = 0
    latencies = []
    for trial, prediction in zip(trials, predictions):
        positive = trial['label'] == 'CONFIRMATION_GESTURE'
        if prediction['detected'] and positive:
            tp += 1
            latencies.append(prediction['latency_ms'])
        elif prediction['detected']:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
    positive_total = tp + fn
    negative_total = tn + fp
    return {'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            'recall': tp / positive_total if positive_total else None,
            'false_positive_rate': fp / negative_total if negative_total else None,
            'mean_latency_ms': sum(latencies) / len(latencies) if latencies else None,
            'median_latency_ms': sorted(latencies)[len(latencies) // 2] if latencies else None}


def split_trials(normal, confirmation, seed=20261115, heldout_fraction=1 / 3):
    rng = random.Random(seed)
    calibration, heldout = [], []
    for group in (normal, confirmation):
        indexes = list(range(len(group)))
        rng.shuffle(indexes)
        held_count = max(1, round(len(group) * heldout_fraction))
        held = set(indexes[:held_count])
        for index, trial in enumerate(group):
            (heldout if index in held else calibration).append(trial)
    return calibration, heldout


def tune(calibration_trials):
    choices = itertools.product(
        (60, 80, 100, 120, 150, 200),
        (20, 35, 50, 70, 100),
        (0.12, 0.20),
        (18, 28, 40),
        (12, 18, 25),
        (3, 5),
        (0.8, 1.5),
    )
    best = None
    for threshold, angle, min_seconds, release, stillness, smoothing, max_gap in choices:
        candidate = replace(WristGestureConfig(), rotation_threshold_dps=threshold,
                            release_threshold_dps=release,
                            minimum_rotation_degrees=angle,
                            minimum_rotation_seconds=min_seconds,
                            stillness_threshold_dps=stillness,
                            smoothing_samples=smoothing,
                            maximum_gap_between_twists_seconds=max_gap)
        predictions = [replay_trial(trial, candidate) for trial in calibration_trials]
        result = metric(calibration_trials, predictions)
        matrix = result['confusion_matrix']
        # First eliminate false cancellations, then recover gestures; use
        # stricter thresholds to break ties without peeking at held-out trials.
        score = (matrix['FP'], -matrix['TP'], -threshold, -angle,
                 -min_seconds, -stillness, -release, smoothing, max_gap)
        if best is None or score < best[0]:
            best = score, candidate, result
    return best[1], best[2]


def trace_trial(trial, config):
    recognizer = WristGestureRecognizer(config)
    previous = None
    for sample in trial['samples']:
        fired = recognizer.feed(sample, confirming=True)
        state = (recognizer.twists, recognizer.sequence_axis, recognizer.rotation_axis,
                 recognizer.first_direction, round(recognizer.rotation_integral, 1),
                 recognizer.rotation_started_ms, recognizer.last_twist_ms)
        if state != previous or fired:
            print(f"trace t={sample.timestamp_ms} state={state} fired={fired}", file=sys.stderr)
        previous = state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal', type=Path, default=Path('data/wrist_normal_20261115_042853.csv'))
    parser.add_argument('--confirmation', type=Path,
                        default=Path('data/wrist_confirmation_20261115_043604.csv'))
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--trace-confirmation-id',
                        help='Print recognizer state transitions for one confirmation trial')
    args = parser.parse_args(argv)
    normal = load_trials(args.normal, 'NORMAL_MOVEMENT')
    confirmation = load_trials(args.confirmation, 'CONFIRMATION_GESTURE')
    if args.trace_confirmation_id:
        trace_trial(next(trial for trial in confirmation
                         if trial['trial_id'] == args.trace_confirmation_id), WristGestureConfig())
    calibration, heldout = split_trials(normal, confirmation, seed=args.seed)
    baseline = WristGestureConfig()
    complete_normal = [trial for trial in normal if trial['trailing_stillness_ms'] >= 1000]
    complete_confirmation = [trial for trial in confirmation
                             if trial['trailing_stillness_ms'] >= 1000 and
                             trial['peak_gyro_magnitude_dps'] >= baseline.rotation_threshold_dps]
    complete_observations = complete_normal + complete_confirmation
    raw_result = metric(normal + confirmation,
                        [replay_trial(trial, baseline) for trial in normal + confirmation])
    complete_result = metric(complete_observations,
                             [replay_trial(trial, baseline) for trial in complete_observations])
    tuning_result = {'status': 'not run',
                     'reason': ('Need at least 9 complete trials in each class for a disjoint calibration/held-out split; '
                                f'found {len(complete_confirmation)} complete confirmation and '
                                f'{len(complete_normal)} complete normal trials.')}
    if min(len(complete_normal), len(complete_confirmation)) >= 9:
        calibration, heldout = split_trials(complete_normal, complete_confirmation, seed=args.seed)
        baseline_calibration = metric(calibration,
                                      [replay_trial(trial, baseline) for trial in calibration])
        baseline_heldout = metric(heldout,
                                  [replay_trial(trial, baseline) for trial in heldout])
        tuned, tuned_calibration = tune(calibration)
        tuned_heldout = metric(heldout,
                               [replay_trial(trial, tuned) for trial in heldout])
        tuning_result = {
            'status': 'evaluated',
            'calibration': {'baseline': baseline_calibration, 'tuned': tuned_calibration},
            'heldout': {'baseline': baseline_heldout, 'tuned': tuned_heldout},
            'tuned_parameters': asdict(tuned),
        }
    report = {
        'trial_counts': {'independent_normal_recordings': len(normal),
                         'independent_confirmation_recordings': len(confirmation),
                         'complete_normal_with_1s_stillness_tail': len(complete_normal),
                         'complete_confirmation_with_motion_and_1s_tail': len(complete_confirmation)},
        'split_seed': args.seed,
        'duration_ms': {
            'normal_min': min(trial['duration_ms'] for trial in normal),
            'normal_max': max(trial['duration_ms'] for trial in normal),
            'confirmation_min': min(trial['duration_ms'] for trial in confirmation),
            'confirmation_max': max(trial['duration_ms'] for trial in confirmation),
        },
        'baseline': {'parameters': asdict(baseline),
                     'all_captured_labeled_clips': raw_result,
                     'complete_observations_only': complete_result},
        'calibration_evaluation': tuning_result,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
