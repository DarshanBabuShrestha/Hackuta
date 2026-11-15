"""Local capture and reporting for experimental seizure-like motion episodes.

The tracker is driven by Arduino/replay sensor timestamps. It records raw and
gravity-compensated axes, keeps a short pre-trigger buffer, and waits for a
continuous sensor-observed recovery period before finalizing a report.
"""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import uuid

import numpy as np


DEFAULT_HISTORY_PATH = Path(__file__).with_name('.episode_reports.json')


def _locked(method):
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class EpisodeTracker:
    """Capture, summarize, persist, and retrieve the five latest episodes."""

    def __init__(self, pre_event_seconds=5.0, trigger_lookback_seconds=5.0,
                 recovery_seconds=5.0, burst_min_seconds=0.2,
                 burst_hysteresis=0.1, minimum_motion_rms_g=0.12,
                 max_sample_gap_ms=400, sample_rate_hz=50, max_reports=5,
                 history_path=DEFAULT_HISTORY_PATH):
        if (pre_event_seconds < 0 or trigger_lookback_seconds <= 0 or
                recovery_seconds <= 0 or burst_min_seconds < 0 or
                not 0 <= burst_hysteresis < 1 or minimum_motion_rms_g <= 0 or
                max_sample_gap_ms <= 0 or sample_rate_hz <= 0 or max_reports <= 0):
            raise ValueError('Invalid episode tracker settings')
        self.pre_event_seconds = float(pre_event_seconds)
        self.trigger_lookback_seconds = float(trigger_lookback_seconds)
        self.recovery_seconds = float(recovery_seconds)
        self.burst_min_seconds = float(burst_min_seconds)
        self.burst_hysteresis = float(burst_hysteresis)
        self.minimum_motion_rms_g = float(minimum_motion_rms_g)
        self.max_sample_gap_ms = int(max_sample_gap_ms)
        self.sample_rate_hz = int(sample_rate_hz)
        self.max_reports = int(max_reports)
        self.history_path = Path(history_path) if history_path else None
        self._lock = threading.RLock()
        self.history = self._load_history()
        self.current = None
        # Keep enough samples to cover the configurable pre-event interval plus
        # the sustained detector interval needed before its first positive result.
        lookback = self.pre_event_seconds + self.trigger_lookback_seconds + 1
        self.pre_buffer = deque(maxlen=max(2, math.ceil(lookback * self.sample_rate_hz)))
        self.pending_windows = deque()
        self.gravity_window = deque(maxlen=max(2, round(.5 * self.sample_rate_hz)))
        self.last_raw_timestamp_ms = None
        self.elapsed_ms = None
        self.last_entry = None
        self.warnings = set()
        self.last_gap_detected = False

    def _load_history(self):
        if not self.history_path or not self.history_path.is_file():
            return []
        try:
            reports = json.loads(self.history_path.read_text(encoding='utf-8'))
            if isinstance(reports, list):
                return [item for item in reports if isinstance(item, dict)][-self.max_reports:]
        except (OSError, ValueError):
            pass
        return []

    @staticmethod
    def _sample_values(sample):
        if isinstance(sample, dict):
            timestamp = sample.get('timestamp_ms')
            axes = (sample.get('x_g'), sample.get('y_g'), sample.get('z_g'))
        else:
            timestamp = getattr(sample, 'timestamp_ms', None)
            values = getattr(sample, 'values', ())
            axes = values[:3]
        if timestamp is None or len(axes) != 3:
            return None
        try:
            timestamp = int(timestamp)
            axes = tuple(float(value) for value in axes)
        except (TypeError, ValueError):
            return None
        if timestamp < 0 or not all(math.isfinite(value) for value in axes):
            return None
        return timestamp, axes

    def _consume_samples(self, sensor_samples):
        self.last_gap_detected = False
        new_entries = []
        for sample in sensor_samples:
            parsed = self._sample_values(sample)
            if parsed is None:
                continue
            raw_timestamp, axes = parsed
            if self.last_raw_timestamp_ms is None:
                delta = 0
                self.elapsed_ms = raw_timestamp
            else:
                if (raw_timestamp < self.last_raw_timestamp_ms and
                        self.last_raw_timestamp_ms < 2 ** 32 - 60000):
                    backwards_ms = self.last_raw_timestamp_ms - raw_timestamp
                    overlap_limit_ms = round(32 * 1000 / self.sample_rate_hz)
                    if backwards_ms <= overlap_limit_ms:
                        # SerialBridge repeats the overlapping half of each window.
                        continue
                    # A larger backwards jump is a board reboot, not millis wrap.
                    delta = round(1000 / self.sample_rate_hz)
                    self.warnings.add('Arduino sensor clock reset during capture; elapsed time across the reset is incomplete.')
                    self.last_gap_detected = True
                else:
                    delta = (raw_timestamp - self.last_raw_timestamp_ms) % (2 ** 32)
                if delta == 0:
                    continue
                if delta > self.max_sample_gap_ms:
                    self.last_gap_detected = True
                    self.warnings.add('Sensor samples were missing or delayed; the report contains a timestamp gap.')
                    self.pre_buffer.clear()
                    self.pending_windows.clear()
                    if self.current:
                        self.current['recovery_since_elapsed_ms'] = None
                    self.gravity_window.clear()
                self.elapsed_ms += delta
            self.last_raw_timestamp_ms = raw_timestamp
            self.gravity_window.append(axes)
            baseline = np.mean(np.asarray(self.gravity_window, dtype=float), axis=0)
            dynamic = np.asarray(axes, dtype=float) - baseline
            movement = float(np.linalg.norm(dynamic))
            entry = {
                'timestamp_ms': raw_timestamp,
                'elapsed_ms': int(self.elapsed_ms),
                'x_g': axes[0], 'y_g': axes[1], 'z_g': axes[2],
                'dynamic_x_g': float(dynamic[0]),
                'dynamic_y_g': float(dynamic[1]),
                'dynamic_z_g': float(dynamic[2]),
                'movement_g': movement,
            }
            new_entries.append(entry)
            self.pre_buffer.append(entry)
            self.last_entry = entry
        return new_entries

    @_locked
    def observe(self, sensor_samples, *, detected=False, rhythmic=False,
                rhythmic_seconds=0.0, frequency_hz=None,
                frequency_valid=False, motion_rms_g=None, window_rms_g=None,
                spectrum=None, source_type='arduino', onset_timestamp_ms=None):
        """Consume timestamped sensor samples and the latest detector window.

        ``sensor_samples`` may overlap between calls; timestamp deduplication
        ensures each physical reading is saved only once.
        """
        entries = self._consume_samples(sensor_samples)
        if not entries:
            return self.summary()
        now = entries[-1]
        elapsed = now['elapsed_ms']
        gap = self.last_gap_detected
        if len(entries) > 1:
            gap = gap or any((b['elapsed_ms'] - a['elapsed_ms']) > self.max_sample_gap_ms
                             for a, b in zip(entries, entries[1:]))
        try:
            window_rms = float(window_rms_g) if window_rms_g is not None else None
        except (TypeError, ValueError):
            window_rms = None
        try:
            frequency = float(frequency_hz) if frequency_hz is not None else None
        except (TypeError, ValueError):
            frequency = None
        frequency_ok = bool(frequency_valid and rhythmic and frequency is not None and
                            math.isfinite(frequency) and frequency > 0)
        try:
            window_motion = float(motion_rms_g) if motion_rms_g is not None else None
        except (TypeError, ValueError):
            window_motion = None
        valid_window_motion = (window_motion is not None and math.isfinite(window_motion)
                               and window_motion >= self.minimum_motion_rms_g)
        rhythmic_now = bool(rhythmic and valid_window_motion and
                            window_rms is not None and
                            window_rms >= self.minimum_motion_rms_g)
        window = {
            'timestamp_ms': now['timestamp_ms'], 'elapsed_ms': elapsed,
            'movement_rms_g': window_rms,
            'frequency_hz': frequency if frequency_ok else None,
            'frequency_valid': frequency_ok,
            'motion_rms_g': window_motion,
            'spectrum': list(spectrum or []) if isinstance(spectrum, (list, tuple)) else [],
        }
        self.pending_windows.append(window)
        pending_cutoff = elapsed - int((self.pre_event_seconds +
                                        self.trigger_lookback_seconds + 1) * 1000)
        while self.pending_windows and self.pending_windows[0]['elapsed_ms'] < pending_cutoff:
            self.pending_windows.popleft()

        if self.current is None and detected:
            if onset_timestamp_ms is not None:
                try:
                    onset_lag = (now['timestamp_ms'] - int(onset_timestamp_ms)) % (2 ** 32)
                except (TypeError, ValueError):
                    onset_lag = int(max(0.0, rhythmic_seconds) * 1000)
                onset_estimate = elapsed - onset_lag
            else:
                onset_estimate = elapsed - int(max(0.0, rhythmic_seconds) * 1000)
            earliest = self.pre_buffer[0]['elapsed_ms'] if self.pre_buffer else onset_estimate
            onset_estimate = max(earliest, onset_estimate)
            onset_entry = next((point for point in self.pre_buffer
                                if point['elapsed_ms'] >= onset_estimate), now)
            onset_elapsed = onset_entry['elapsed_ms']
            onset_raw = onset_entry['timestamp_ms']
            recording_cutoff = onset_elapsed - int(self.pre_event_seconds * 1000)
            seeded = []
            for point in self.pre_buffer:
                if point['elapsed_ms'] >= recording_cutoff:
                    copied = dict(point)
                    copied['phase'] = 'pre_event' if point['elapsed_ms'] < onset_elapsed else 'episode'
                    seeded.append(copied)
            self.current = {
                'episode_id': uuid.uuid4().hex,
                'source_type': 'simulated' if source_type != 'arduino' else 'arduino',
                'status': 'EPISODE_ACTIVE',
                'start_timestamp_ms': onset_raw,
                'start_elapsed_ms': onset_elapsed,
                'detection_timestamp_ms': now['timestamp_ms'],
                'samples': seeded,
                'frequency_timeline': [],
                'last_active_timestamp_ms': onset_raw,
                'last_active_elapsed_ms': onset_elapsed,
                'recovery_since_elapsed_ms': None,
                'recovery_end_timestamp_ms': None,
                'warnings': set(),
                'peak_window_rms_g': None,
                'peak_movement_timestamp_ms': None,
                'peak_spectrum': None,
                'peak_spectrum_timestamp_ms': None,
                'missing_gap': False,
            }
            if not seeded or seeded[0]['elapsed_ms'] > recording_cutoff + 40:
                self.current['warnings'].add('The requested pre-event buffer was not fully available.')
            self.current['warnings'].update(self.warnings)
            self.warnings.clear()
            for item in self.pending_windows:
                if item['elapsed_ms'] < onset_elapsed:
                    continue
                self._consider_peak_window(item)
                self.current['frequency_timeline'].append(
                    {key: value for key, value in item.items() if key != 'spectrum'})

        if self.current is None:
            return self.summary()

        current = self.current
        if gap:
            current['missing_gap'] = True
            current['warnings'].add('Sensor samples were missing or delayed; the report contains a timestamp gap.')
            current['recovery_since_elapsed_ms'] = None
        current['warnings'].update(self.warnings)
        self.warnings.clear()
        if gap and current['frequency_timeline']:
            previous_frequency = current['frequency_timeline'][-1]
            if previous_frequency.get('frequency_valid'):
                current['frequency_timeline'].append({
                    'timestamp_ms': previous_frequency['timestamp_ms'],
                    'elapsed_ms': previous_frequency['elapsed_ms'],
                    'movement_rms_g': None, 'motion_rms_g': None,
                    'frequency_hz': None, 'frequency_valid': False,
                    'data_gap': True,
                })
        for point in entries:
            if point['elapsed_ms'] <= current['samples'][-1]['elapsed_ms']:
                continue
            copied = dict(point)
            copied['phase'] = 'episode' if point['elapsed_ms'] <= current['last_active_elapsed_ms'] else 'recovery'
            current['samples'].append(copied)

        if not current['frequency_timeline'] or current['frequency_timeline'][-1]['elapsed_ms'] < elapsed:
            current['frequency_timeline'].append(
                {key: value for key, value in window.items() if key != 'spectrum'})
        if rhythmic_now:
            current['last_active_timestamp_ms'] = now['timestamp_ms']
            current['last_active_elapsed_ms'] = elapsed
            current['recovery_since_elapsed_ms'] = None
            current['status'] = 'EPISODE_ACTIVE'
            for point in current['samples']:
                if point['elapsed_ms'] >= current['start_elapsed_ms'] and point['elapsed_ms'] <= elapsed:
                    point['phase'] = 'episode'
        else:
            if current['recovery_since_elapsed_ms'] is None:
                current['recovery_since_elapsed_ms'] = elapsed
            current['status'] = 'RECOVERING'
            recovery_ms = elapsed - current['recovery_since_elapsed_ms']
            if recovery_ms >= self.recovery_seconds * 1000 and not gap:
                current['recovery_end_timestamp_ms'] = now['timestamp_ms']
                return self._finalize()
        if rhythmic_now or elapsed <= current['last_active_elapsed_ms']:
            self._consider_peak_window(window)
        return self.summary()

    def _consider_peak_window(self, window):
        current = self.current
        intensity = window.get('movement_rms_g')
        if intensity is None or not math.isfinite(intensity):
            return
        if current['peak_window_rms_g'] is None or intensity > current['peak_window_rms_g']:
            current['peak_window_rms_g'] = intensity
            current['peak_movement_timestamp_ms'] = window['timestamp_ms']
            current['peak_movement_elapsed_ms'] = window['elapsed_ms']
            current['peak_spectrum'] = window.get('spectrum') or None
            current['peak_spectrum_timestamp_ms'] = (window['timestamp_ms']
                                                       if window.get('spectrum') else None)

    def _current_report(self, status=None, end_timestamp_ms=None, end_elapsed_ms=None,
                        include_series=True):
        current = self.current
        if current is None:
            return None
        end_elapsed_ms = (current['last_active_elapsed_ms'] if end_elapsed_ms is None
                          else end_elapsed_ms)
        episode_samples = [point for point in current['samples']
                           if current['start_elapsed_ms'] <= point['elapsed_ms'] <= end_elapsed_ms]
        magnitudes = [point['movement_g'] for point in episode_samples]
        avg_rms = (float(np.sqrt(np.mean(np.square(magnitudes)))) if magnitudes else None)
        timeline = [item for item in current['frequency_timeline']
                    if current['start_elapsed_ms'] <= item['elapsed_ms'] <= end_elapsed_ms]
        valid_frequencies = [item['frequency_hz'] for item in timeline
                             if item.get('frequency_valid') and item.get('frequency_hz') is not None]
        bursts, above_seconds = self._above_average(magnitudes, episode_samples, avg_rms)
        peak = current['peak_window_rms_g']
        report = {
            'episode_id': current['episode_id'],
            'status': status or current['status'],
            'source_type': current['source_type'],
            'sample_rate_hz': self.sample_rate_hz,
            'start_timestamp_ms': current['start_timestamp_ms'],
            'start_elapsed_ms': current['start_elapsed_ms'],
            'end_timestamp_ms': (current['last_active_timestamp_ms'] if end_timestamp_ms is None
                                 else end_timestamp_ms),
            'end_elapsed_ms': end_elapsed_ms,
            'detection_timestamp_ms': current['detection_timestamp_ms'],
            'recovery_end_timestamp_ms': current['recovery_end_timestamp_ms'],
            'duration_seconds': max(0.0, (end_elapsed_ms - current['start_elapsed_ms']) / 1000),
            'average_frequency_hz': (float(np.mean(valid_frequencies)) if valid_frequencies else None),
            'highest_dominant_frequency_hz': (max(valid_frequencies) if valid_frequencies else None),
            'frequency_variability_hz': (float(np.std(valid_frequencies)) if valid_frequencies else None),
            'valid_frequency_windows': len(valid_frequencies),
            'peak_movement_g_rms': peak,
            'average_movement_g_rms': avg_rms,
            'above_average_bursts': bursts,
            'time_above_average_seconds': above_seconds,
            'peak_movement_timestamp_ms': current['peak_movement_timestamp_ms'],
            'peak_movement_elapsed_ms': current.get('peak_movement_elapsed_ms'),
            'sample_count': len(current['samples']),
            'missing_data': current['missing_gap'],
            'data_quality_warnings': sorted(current['warnings']),
            'movement_intensity_method': ('Dynamic vector RMS from signed X/Y/Z acceleration after subtracting a trailing 0.5-second per-axis mean; peak is the highest valid 0.64-second detector-window RMS. Pre-event samples and the five-second recovery tail are excluded from episode summary metrics.'),
            'frequency_method': 'Frequency samples are retained only when the seizure-like detector reports rhythmic movement and its motion and spectral gates pass; invalid windows remain null gaps.',
            'recording_start_timestamp_ms': (current['samples'][0]['timestamp_ms'] if current['samples'] else None),
            'recording_start_elapsed_ms': (current['samples'][0]['elapsed_ms'] if current['samples'] else None),
            'recording_end_timestamp_ms': (current['samples'][-1]['timestamp_ms'] if current['samples'] else None),
            'recording_end_elapsed_ms': (current['samples'][-1]['elapsed_ms'] if current['samples'] else None),
            'peak_spectrum_timestamp_ms': current['peak_spectrum_timestamp_ms'],
        }
        if include_series:
            report['samples'] = [dict(point) for point in current['samples']]
            report['frequency_timeline'] = [dict(item) for item in current['frequency_timeline']]
            report['peak_spectrum'] = current['peak_spectrum']
        return report

    def _above_average(self, magnitudes, samples, average):
        if average is None or not samples:
            return 0, 0.0
        intervals = []
        start = None
        stop_below = average * (1 - self.burst_hysteresis)
        previous = None
        for point, magnitude in zip(samples, magnitudes):
            ts = point['elapsed_ms']
            if previous is not None and ts - previous > self.max_sample_gap_ms:
                if start is not None:
                    intervals.append((start, previous))
                start = None
            if start is None and magnitude > average:
                start = ts
            elif start is not None and magnitude < stop_below:
                intervals.append((start, ts))
                start = None
            previous = ts
        if start is not None and previous is not None:
            intervals.append((start, previous))
        valid = [(a, b) for a, b in intervals if (b - a) / 1000 >= self.burst_min_seconds]
        return len(valid), sum((b - a) / 1000 for a, b in valid)

    def _finalize(self):
        report = self._current_report(status='REPORT_READY')
        report['created_at'] = datetime.now(timezone.utc).isoformat()
        self.history = [item for item in self.history if item.get('episode_id') != report['episode_id']]
        self.history.append(report)
        self.history = self.history[-self.max_reports:]
        self._persist()
        self.current = None
        return self.summary(completed=report)

    def _persist(self):
        if not self.history_path:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(self.history_path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.history, separators=(',', ':')), encoding='utf-8')
        os.replace(temporary, self.history_path)

    def summary(self, completed=None):
        with self._lock:
            if self.current:
                report = self._current_report(include_series=False)
                report['status'] = self.current['status']
                return self._public_summary(report)
            report = completed or (self.history[-1] if self.history else None)
            if report is None:
                return None
            return self._public_summary(report)

    @staticmethod
    def _public_summary(report):
        return {key: value for key, value in report.items()
                if key not in ('samples', 'frequency_timeline', 'peak_spectrum')}

    def history_summaries(self):
        with self._lock:
            return [self._public_summary(report) for report in reversed(self.history)]

    def get_report(self, episode_id):
        with self._lock:
            if self.current and self.current['episode_id'] == episode_id:
                return self._current_report()
            return next((report for report in self.history
                         if report.get('episode_id') == episode_id), None)

    def mark_disconnected(self, status='Sensor disconnected'):
        with self._lock:
            self.pre_buffer.clear()
            self.pending_windows.clear()
            self.gravity_window.clear()
            if self.current:
                warning = f'{status}; episode recording has a gap and is not considered ended.'
                self.current['missing_gap'] = True
                self.current['warnings'].add(warning)
                self.current['recovery_since_elapsed_ms'] = None
                self.current['status'] = 'DATA_GAP'
                timeline = self.current['frequency_timeline']
                if timeline and timeline[-1].get('frequency_valid'):
                    last = timeline[-1]
                    timeline.append({'timestamp_ms': last['timestamp_ms'],
                                     'elapsed_ms': last['elapsed_ms'],
                                     'movement_rms_g': None, 'motion_rms_g': None,
                                     'frequency_hz': None, 'frequency_valid': False,
                                     'data_gap': True})
            else:
                self.warnings.add(f'{status} before an episode; pre-event capture may be incomplete.')

    def mark_reconnected(self):
        with self._lock:
            self.gravity_window.clear()
            if self.current:
                self.current['recovery_since_elapsed_ms'] = None
                self.current['status'] = 'EPISODE_ACTIVE'

