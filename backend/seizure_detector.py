"""Experimental simulated-seizure-like movement classifier.

This classifier is not clinically validated. Its model was trained on 206-sample,
three-axis UEA Epilepsy wrist accelerometer recordings at 16 Hz. Arduino axes,
units, placement, and filtering may differ from training data.
"""
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from ml.seizure.train_epilepsy_models import features

DEFAULT_MODEL = (Path(__file__).resolve().parents[1] / 'ml' / 'seizure' /
                 'seizure_models' / 'epilepsy_rf.joblib')
WINDOW_SIZE = 206
SAMPLE_PERIOD_MS = 62.5


class SeizureMotionDetector:
    """Classify downsampled acceleration windows from overlapping IMU windows."""

    def __init__(self, model_path=DEFAULT_MODEL, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid threshold')
        artifact = joblib.load(model_path)
        if (artifact.get('feature_length') != 44 or
                artifact.get('window_size') != WINDOW_SIZE or
                artifact.get('input_axes') != 3 or
                artifact.get('feature_extraction') != 'train_epilepsy_models.features'):
            raise ValueError('Epilepsy model feature contract mismatch')
        self.model = artifact['model']
        if 1 not in self.model.classes_:
            raise ValueError('Model lacks simulated seizure-like motion class')
        self.threshold = threshold
        self.samples = deque(maxlen=WINDOW_SIZE)
        self.last_index = -1
        self.next_sample_ms = None
        self.last_probability = 0.0

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
            if self.next_sample_ms is None:
                self.next_sample_ms = timestamp
            if timestamp + 1e-6 >= self.next_sample_ms:
                self.samples.append(row[:3].copy())
                self.next_sample_ms += SAMPLE_PERIOD_MS
        self.last_index += len(new)
        if len(self.samples) < WINDOW_SIZE:
            return None
        matrix = np.asarray(self.samples, dtype=np.float32)[None, ...]
        vector = features(matrix)
        probabilities = self.model.predict_proba(vector)[0]
        self.last_probability = float(probabilities[list(self.model.classes_).index(1)])
        label = 'simulated_seizure_like_motion' if self.last_probability >= self.threshold else 'normal_activity'
        return label, self.last_probability
