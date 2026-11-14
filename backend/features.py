"""Baseline feature contract applied to 32-sample, six-axis motion windows.

Wire units: acceleration g, angular velocity degrees/second.
Model units: acceleration m/s², angular velocity radians/second.
"""
import numpy as np
from ml.train_odyssey_baseline import extract_features as baseline_extract_features
from ml.train_odyssey_baseline import feature_names as baseline_feature_names

WINDOW_SIZE = 32
G = 9.80665
FEATURE_NAMES = tuple(baseline_feature_names())


def wire_to_si(values):
    array = np.asarray(values, dtype=float)
    if array.shape[-1] != 6 or not np.all(np.isfinite(array)):
        raise ValueError('Expected six finite IMU values')
    result = array.copy()
    result[..., :3] *= G
    result[..., 3:] *= np.pi / 180.0
    return result


def extract_features(si_window):
    """Return the 56 named features in model order from an SI-unit window."""
    data = np.asarray(si_window, dtype=float)
    if data.shape != (WINDOW_SIZE, 6) or not np.all(np.isfinite(data)):
        raise ValueError(f'Expected {WINDOW_SIZE} finite six-axis readings')
    result = baseline_extract_features({'acg': data[:, :3], 'gyro': data[:, 3:]})
    if result is None:
        raise ValueError('Baseline feature extractor rejected window')
    return result


def extract_wire_features(wire_window):
    return extract_features(wire_to_si(wire_window))
