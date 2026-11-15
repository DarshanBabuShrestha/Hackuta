"""Window classifier and cooldown."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, WINDOW_SIZE, extract_wire_features

LOG = logging.getLogger(__name__)
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'ml' / 'odyssey_window_model.joblib'


class FallDetector:
    def __init__(self, model_path=DEFAULT_MODEL, threshold=0.7, cooldown=20, clock=time.monotonic):
        if not 0 <= threshold <= 1 or cooldown < 0:
            raise ValueError('Invalid threshold or cooldown')
        artifact = joblib.load(model_path)
        if artifact.get('feature_names') != FEATURE_NAMES or artifact.get('window_size') != WINDOW_SIZE:
            raise ValueError('Model feature contract mismatch')
        self.model = artifact['model']
        if 'fall' not in self.model.classes_:
            raise ValueError('Model lacks fall class')
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.last_trigger = float('-inf')

    def predict(self, wire_window):
        features = extract_wire_features(wire_window).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        probability = float(probabilities[list(self.model.classes_).index('fall')])
        label = 'fall' if probability >= self.threshold else 'adl'
        now = self.clock()
        trigger = label == 'fall' and now - self.last_trigger >= self.cooldown
        if trigger:
            self.last_trigger = now
        LOG.info('Fall class: %s; trigger: %s', label, trigger)
        return label, probability, trigger
