"""Experimental simulated-seizure-like movement classifier.

This classifier is not clinically validated. Its model was trained on 206-sample,
three-axis UEA Epilepsy wrist accelerometer recordings at 16 Hz. Arduino axes,
units, placement, and filtering may differ from training data.
"""
from collections import deque
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import periodogram

from ml.seizure.train_epilepsy_models import features

DEFAULT_MODEL = (Path(__file__).resolve().parents[1] / 'ml' / 'seizure' /
                 'seizure_models' / 'epilepsy_rf.joblib')
WINDOW_SIZE = 206
SAMPLE_PERIOD_MS = 62.5
SENSOR_RATE_HZ = 50
RHYTHMIC_SECONDS = 5.0
MIN_MOTION_RMS_G = 0.12
MIN_PEAK_POWER_SHARE = 0.08
MIN_BAND_POWER_SHARE = 0.50
LOW_HZ = 3.0
HIGH_HZ = 8.0
DEFAULT_THRESHOLD = 0.35


class SeizureMotionDetector:
    """Classify downsampled acceleration windows from overlapping IMU windows."""

    def __init__(self, model_path=DEFAULT_MODEL, motion_rms_g=MIN_MOTION_RMS_G,
                 peak_power_share=MIN_PEAK_POWER_SHARE, low_hz=LOW_HZ,
                 high_hz=HIGH_HZ, rhythmic_seconds=RHYTHMIC_SECONDS,
                 threshold=DEFAULT_THRESHOLD,
                 band_power_share=MIN_BAND_POWER_SHARE):
        if (motion_rms_g <= 0 or not 0 < peak_power_share <= 1 or
                not 0 < band_power_share <= 1 or
                not 0 < low_hz < high_hz < SENSOR_RATE_HZ / 2 or
                rhythmic_seconds <= 0 or not 0 <= threshold <= 1):
            raise ValueError('Invalid seizure motion verification threshold')
        self.motion_window_size = math.ceil(rhythmic_seconds * SENSOR_RATE_HZ)
        fft_bins = np.fft.rfftfreq(self.motion_window_size, 1 / SENSOR_RATE_HZ)
        if not np.any((fft_bins >= low_hz) & (fft_bins <= high_hz)):
            raise ValueError('FFT band must contain a frequency bin')
        artifact = joblib.load(model_path)
        if (artifact.get('feature_length') != 44 or
                artifact.get('window_size') != WINDOW_SIZE or
                artifact.get('input_axes') != 3 or
                artifact.get('feature_extraction') != 'train_epilepsy_models.features'):
            raise ValueError('Epilepsy model feature contract mismatch')
        self.model = artifact['model']
        if 1 not in self.model.classes_:
            raise ValueError('Model lacks simulated seizure-like motion class')
        self.samples = deque(maxlen=WINDOW_SIZE)
        self.motion_samples = deque(maxlen=self.motion_window_size)
        self.motion_rms_g = motion_rms_g
        self.peak_power_share = peak_power_share
        self.band_power_share = band_power_share
        self.low_hz, self.high_hz = low_hz, high_hz
        self.rhythmic_seconds = rhythmic_seconds
        self.threshold = threshold
        self.status = {'motion': 'collecting', 'rhythmic_seconds': 0.0,
                       'motion_rms_g': 0.0, 'peak_hz': 0.0,
                       'peak_power_share': 0.0, 'band_power_share': 0.0,
                       'reason': 'collecting motion samples',
                       'probability': None}
        self.last_index = -1
        self.next_sample_ms = None
        self.last_probability = 0.0

    def _verify_motion(self):
        """Verify activity throughout the rolling window and its FFT rhythm."""
        if len(self.motion_samples) < self.motion_window_size:
            return
        acceleration = np.asarray(self.motion_samples, dtype=float)
        dynamic = acceleration - acceleration.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(dynamic * dynamic, axis=1))))
        # A short burst cannot stand in for five seconds of movement.
        block_rms = []
        for start in range(0, len(acceleration), SENSOR_RATE_HZ):
            block = acceleration[start:start + SENSOR_RATE_HZ]
            centered = block - block.mean(axis=0)
            block_rms.append(float(np.sqrt(np.mean(np.sum(centered * centered,
                                                           axis=1)))))
        active = min(block_rms) >= self.motion_rms_g
        frequencies, power = periodogram(dynamic, fs=SENSOR_RATE_HZ,
                                         window='hann', scaling='spectrum', axis=0)
        power = power.sum(axis=1)
        band = (frequencies >= self.low_hz) & (frequencies <= self.high_hz)
        peak = int(np.argmax(np.where(band, power, -1)))
        total = power[1:].sum() + 1e-12
        share = float(power[peak] / total)
        band_share = float(power[band].sum() / total)
        peak_hz = float(frequencies[peak])
        rhythmic = (active and share >= self.peak_power_share and
                    band_share >= self.band_power_share)
        duration = self.motion_window_size / SENSOR_RATE_HZ if rhythmic else 0.0
        reason = ('stationary' if not active else
                  'non-rhythmic movement' if not rhythmic else
                  'motion verification passed')
        self.status.update(motion='stationary' if not active else
                           'rhythmic' if rhythmic else 'ordinary movement',
                           rhythmic_seconds=duration, motion_rms_g=rms,
                           peak_hz=peak_hz, peak_power_share=share,
                           band_power_share=band_share, reason=reason)

    def push_window(self, wire_window, window_end_ms, hop=16):
        """Consume each new sample once; return latest probability when ready.

        Input windows overlap by ``hop`` samples and arrive at 50 Hz. Acceleration
        is kept in wire units (g); model transfer validity requires separate
        calibration against the training dataset.
        """
        data = np.asarray(wire_window, dtype=float)
        if data.ndim != 2 or data.shape[1] != 6 or not np.all(np.isfinite(data)):
            raise ValueError('Expected finite six-axis IMU window')
        # The first complete window supplies its samples from its start. Later
        # windows contribute only the non-overlapping tail.
        if self.last_index < 0:
            new = data
            first_ms = window_end_ms - (len(data) - 1) * 20
        else:
            new = data[-hop:]
            first_ms = window_end_ms - (hop - 1) * 20
        for i, row in enumerate(new):
            timestamp = first_ms + i * 20
            self.motion_samples.append(row[:3].copy())
            if self.next_sample_ms is None:
                self.next_sample_ms = timestamp
            if timestamp + 1e-6 >= self.next_sample_ms:
                self.samples.append(row[:3].copy())
                self.next_sample_ms += SAMPLE_PERIOD_MS
        self._verify_motion()
        self.last_index += len(new)
        if len(self.samples) < WINDOW_SIZE:
            if self.status['reason'] == 'motion verification passed':
                self.status['reason'] = 'collecting 206-sample ML window'
            return None
        matrix = np.asarray(self.samples, dtype=np.float32)[None, ...]
        vector = features(matrix)
        probabilities = self.model.predict_proba(vector)[0]
        self.last_probability = float(probabilities[list(self.model.classes_).index(1)])
        self.status['probability'] = self.last_probability
        motion_passed = (self.status['motion'] == 'rhythmic' and
                         self.status['rhythmic_seconds'] >= self.rhythmic_seconds)
        score_passed = self.last_probability >= self.threshold
        if motion_passed and score_passed:
            self.status['reason'] = 'motion and ML confirmation passed'
            label = 'possible_seizure_like_movement'
        else:
            if motion_passed:
                self.status['reason'] = 'ML probability below threshold'
            label = 'normal_activity'
        return label, self.last_probability
