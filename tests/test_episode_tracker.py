"""Episode recording uses unique timestamped sensor samples and a sensor-timed recovery."""
import csv
import io
import json
import math
import queue

import numpy as np
from fastapi.testclient import TestClient

from backend.dashboard import Dashboard
from backend.episode_tracker import EpisodeTracker
from backend.serial_bridge import Sample


def samples_for(seconds=21, intervals=((5, 13),)):
    count = int(seconds * 50)
    times = np.arange(count) / 50
    x = np.zeros(count)
    for start, end in intervals:
        active = (times >= start) & (times < end)
        x[active] = .6 * np.sin(2 * np.pi * 3 * times[active])
    return [Sample(i * 20, (float(x[i]), 0.0, 1.0, 0.0, 0.0, 0.0))
            for i in range(count)]


def feed(tracker, sensor, *, detected_at=10.0, rhythmic_intervals=((5, 13),),
         source_type='arduino'):
    last = None
    for end in range(31, len(sensor), 16):
        window = sensor[end - 31:end + 1]
        timestamp = window[-1].timestamp_ms
        seconds = timestamp / 1000
        rhythmic = any(start <= seconds < stop for start, stop in rhythmic_intervals)
        axes = np.asarray([sample.values[:3] for sample in window])
        centered = axes - axes.mean(axis=0)
        window_rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        tracker.observe(
            window,
            detected=detected_at <= seconds < detected_at + .4,
            rhythmic=rhythmic,
            rhythmic_seconds=5.0 if rhythmic else 0.0,
            frequency_hz=3.0 if seconds < 10.5 else 5.0,
            frequency_valid=rhythmic,
            motion_rms_g=.35 if rhythmic else .0,
            window_rms_g=window_rms,
            spectrum=[{'hz': 3.0, 'power': .2}, {'hz': 5.0, 'power': .8}],
            source_type=source_type,
            onset_timestamp_ms=5000 if detected_at <= seconds else None,
        )
        last = tracker.summary()
    return last


def test_full_waveform_pre_event_frequency_and_report_persist(tmp_path):
    path = tmp_path / 'reports.json'
    tracker = EpisodeTracker(pre_event_seconds=2, trigger_lookback_seconds=5,
                             recovery_seconds=5, minimum_motion_rms_g=.05,
                             history_path=path)
    feed(tracker, samples_for(), detected_at=10.0)

    assert tracker.current is None
    report = tracker.history[-1]
    assert report['status'] == 'REPORT_READY'
    assert report['source_type'] == 'arduino'
    assert report['start_timestamp_ms'] == 5000
    assert math.isclose(report['duration_seconds'], 7.8, abs_tol=.4)
    assert report['recovery_end_timestamp_ms'] - report['end_timestamp_ms'] >= 5000
    assert report['sample_count'] > 500
    assert report['recording_start_timestamp_ms'] == 3000
    assert report['recording_end_timestamp_ms'] >= 18000
    assert len({sample['timestamp_ms'] for sample in report['samples']}) == len(report['samples'])
    assert report['average_frequency_hz'] is not None
    assert report['highest_dominant_frequency_hz'] == 5.0
    assert report['frequency_variability_hz'] > 0
    assert report['peak_movement_g_rms'] > 0
    assert report['peak_movement_timestamp_ms'] is not None
    assert report['peak_spectrum']
    assert any(point['frequency_hz'] is None for point in report['frequency_timeline'])
    assert json.loads(path.read_text(encoding='utf-8'))[0]['episode_id'] == report['episode_id']


def test_overlapping_windows_do_not_duplicate_and_resume_keeps_same_episode(tmp_path):
    tracker = EpisodeTracker(pre_event_seconds=1, trigger_lookback_seconds=5,
                             recovery_seconds=5, minimum_motion_rms_g=.05,
                             history_path=tmp_path / 'episodes.json')
    signal = samples_for(seconds=24, intervals=((5, 8), (10, 13)))
    feed(tracker, signal, detected_at=7.0, rhythmic_intervals=((5, 8), (10, 13)))

    assert tracker.current is None
    assert len(tracker.history) == 1
    report = tracker.history[0]
    assert report['duration_seconds'] >= 7.5
    assert report['duration_seconds'] < 8.2
    recorded = [sample['timestamp_ms'] for sample in report['samples']]
    assert len(recorded) == len(set(recorded))


def test_disconnect_marks_incomplete_and_does_not_end_episode(tmp_path):
    tracker = EpisodeTracker(pre_event_seconds=1, trigger_lookback_seconds=5,
                             recovery_seconds=1, minimum_motion_rms_g=.05,
                             history_path=tmp_path / 'episodes.json')
    signal = samples_for(seconds=12, intervals=((5, 12),))
    for end in range(31, 501, 16):
        window = signal[end - 31:end + 1]
        seconds = window[-1].timestamp_ms / 1000
        axes = np.asarray([sample.values[:3] for sample in window])
        centered = axes - axes.mean(axis=0)
        tracker.observe(
            window, detected=seconds >= 7 and seconds < 7.4,
            rhythmic=5 <= seconds < 12, rhythmic_seconds=5,
            frequency_hz=3, frequency_valid=True, motion_rms_g=.3,
            window_rms_g=float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
            onset_timestamp_ms=5000,
        )
    tracker.mark_disconnected('Serial disconnected')
    assert tracker.current is not None
    assert tracker.current['status'] == 'DATA_GAP'
    assert tracker.current['missing_gap']
    assert tracker.history == []
    tracker.mark_reconnected()
    quiet = [Sample(10000 + i * 20, (0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
             for i in range(120)]
    for end in range(31, len(quiet), 16):
        window = quiet[end - 31:end + 1]
        tracker.observe(window, rhythmic=False, frequency_valid=False,
                        motion_rms_g=0.0, window_rms_g=0.0)
    assert tracker.current is None
    assert tracker.history[-1]['missing_data']
    assert any('disconnected' in warning.lower()
               for warning in tracker.history[-1]['data_quality_warnings'])


def test_above_average_bursts_use_hysteresis_and_minimum_duration(tmp_path):
    tracker = EpisodeTracker(burst_min_seconds=.2, burst_hysteresis=.1,
                             history_path=tmp_path / 'episodes.json')
    magnitudes = [.2] * 16 + [.1] * 6 + [.2] * 16 + [.1] * 20 + [.2] * 5
    points = [{'elapsed_ms': index * 20} for index in range(len(magnitudes))]
    bursts, seconds = tracker._above_average(magnitudes, points, average=.15)
    assert bursts == 2
    assert math.isclose(seconds, .64, abs_tol=.02)


def test_report_json_and_csv_exports_include_full_timestamped_series(tmp_path):
    tracker = EpisodeTracker(pre_event_seconds=1, trigger_lookback_seconds=5,
                             recovery_seconds=1, minimum_motion_rms_g=.05,
                             history_path=tmp_path / 'episodes.json')
    feed(tracker, samples_for(seconds=16, intervals=((5, 10),)), detected_at=7.0,
         source_type='simulated')
    report_id = tracker.history[-1]['episode_id']
    dashboard = Dashboard(queue.Queue(), 'COM9', episode_tracker=tracker)
    client = TestClient(dashboard.app, client=('127.0.0.1', 54321))

    json_response = client.get(f'/api/episodes/{report_id}/export?format=json')
    csv_response = client.get(f'/api/episodes/{report_id}/export?format=csv')
    assert json_response.status_code == 200
    assert json_response.json()['samples']
    assert json_response.json()['source_type'] == 'simulated'
    assert csv_response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert rows and rows[0]['record_type'] == 'sensor_sample'
    assert rows[0]['elapsed_ms']
    assert any(row['frequency_hz'] for row in rows)


def test_history_is_limited_to_five_reports(tmp_path):
    tracker = EpisodeTracker(pre_event_seconds=0, trigger_lookback_seconds=1,
                             recovery_seconds=.2, minimum_motion_rms_g=.01,
                             max_reports=5, history_path=tmp_path / 'episodes.json')
    for episode in range(6):
        onset = episode * 12000
        values = []
        for i in range(120):
            timestamp = onset + i * 20
            x = .5 * math.sin(2 * math.pi * 3 * i / 50) if i < 60 else 0.0
            values.append(Sample(timestamp, (x, 0, 1, 0, 0, 0)))
            if len(values) >= 32 and (i - 31) % 16 == 0:
                ts = values[-1].timestamp_ms
                tracker.observe(values[-32:], detected=(i == 31), rhythmic=i < 60,
                                rhythmic_seconds=1, frequency_hz=3,
                                frequency_valid=i < 60, motion_rms_g=.2,
                                window_rms_g=.2 if i < 60 else 0,
                                onset_timestamp_ms=onset)
    assert len(tracker.history) <= 5
