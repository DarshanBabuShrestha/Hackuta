import csv

import pytest

from scripts.evaluate_wrist_gesture import load_trials, split_trials


FIELDS = ('timestamp_ms', 'label', 'movement_id', 'instruction',
          'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps')


def write_trial(path, trial_id, start_ms, label='NORMAL_MOVEMENT', count=120):
    with path.open('a', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if path.stat().st_size == 0:
            writer.writeheader()
        for index in range(count):
            writer.writerow({'timestamp_ms': start_ms + index * 20, 'label': label,
                             'movement_id': trial_id, 'instruction': 'test trial',
                             'ax_g': 0, 'ay_g': 0, 'az_g': 1,
                             'gx_dps': 0, 'gy_dps': 0, 'gz_dps': 0})


def test_load_trials_groups_rows_and_validates_capture_duration(tmp_path):
    recording = tmp_path / 'normal.csv'
    write_trial(recording, '1', 1000)
    write_trial(recording, '2', 7000)
    trials = load_trials(recording, 'NORMAL_MOVEMENT')
    assert len(trials) == 2
    assert [trial['trial_id'] for trial in trials] == ['1', '2']
    assert all(len(trial['samples']) == 120 for trial in trials)
    assert all(trial['duration_ms'] == 2380 for trial in trials)
    assert all(trial['trailing_stillness_ms'] >= 1000 for trial in trials)


def test_loader_rejects_mixed_or_unexpected_trial_labels(tmp_path):
    recording = tmp_path / 'bad.csv'
    write_trial(recording, '1', 1000, label='NORMAL_MOVEMENT')
    with pytest.raises(ValueError, match='unexpected/mixed labels'):
        load_trials(recording, 'CONFIRMATION_GESTURE')


def test_calibration_and_heldout_splits_never_split_a_trial():
    normal = [{'trial_id': str(i), 'label': 'NORMAL_MOVEMENT'} for i in range(9)]
    confirmation = [{'trial_id': str(i), 'label': 'CONFIRMATION_GESTURE'} for i in range(9)]
    calibration, heldout = split_trials(normal, confirmation, seed=12)
    calibration_keys = {(trial['label'], trial['trial_id']) for trial in calibration}
    heldout_keys = {(trial['label'], trial['trial_id']) for trial in heldout}
    assert calibration_keys.isdisjoint(heldout_keys)
    assert len(calibration) + len(heldout) == 18
    assert len(calibration) == 12
    assert len(heldout) == 6
