"""Experimental tremor-like rhythmic motion detector for 50 Hz IMU input."""
from collections import deque

import numpy as np
from scipy.signal import welch

WINDOW_SIZE = 250  # Five seconds at the nominal 50 Hz sensor rate.
SAMPLE_RATE_HZ = 50
LOW_HZ = 3.0
HIGH_HZ = 12.0
MIN_BAND_RMS_G = 0.08
MIN_BAND_POWER_SHARE = 0.35


class TremorMotionDetector:
    """Flag sustained, rhythmic acceleration in a broad tremor-like band.

    This is a motion heuristic, not a medical tremor detector. Sensor placement
    and thresholds have not been validated for wearer-specific use.
    """

    def __init__(self, min_band_rms_g=MIN_BAND_RMS_G,
                 min_band_power_share=MIN_BAND_POWER_SHARE):
        self.min_band_rms_g = min_band_rms_g
        self.min_band_power_share = min_band_power_share
        self.samples = deque(maxlen=WINDOW_SIZE)
        self.seen_window = False
        self.last_spectrum = []

    def spectrum(self):
        return self.last_spectrum

    def push_window(self, wire_window, hop=16):
        """Consume overlapping 32-row six-axis windows from SerialBridge."""
        data = np.asarray(wire_window, dtype=float)
        if data.ndim != 2 or data.shape[1] != 6 or not np.all(np.isfinite(data)):
            raise ValueError('Expected finite six-axis IMU window')
        new = data if not self.seen_window else data[-hop:]
        self.seen_window = True
        self.samples.extend(row[:3].copy() for row in new)
        if len(self.samples) < WINDOW_SIZE:
            return None

        acceleration = np.asarray(self.samples, dtype=float)
        magnitude = np.linalg.norm(acceleration, axis=1)
        signal = magnitude - magnitude.mean()
        frequencies, power = welch(signal, fs=SAMPLE_RATE_HZ, nperseg=WINDOW_SIZE)
        band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
        if not np.any(band):
            return 'normal_activity', 0.0, 0.0

        resolution = frequencies[1] - frequencies[0]
        band_power = float(np.sum(power[band]) * resolution)
        total_power = float(np.sum(power) * resolution)
        band_rms = float(np.sqrt(band_power))
        power_share = band_power / (total_power + 1e-12)
        peak_frequency = float(frequencies[band][np.argmax(power[band])])
        self.last_spectrum = [
            {'hz': round(float(frequency), 2), 'power': round(float(value), 8)}
            for frequency, value in zip(frequencies[band], power[band])
        ]
        detected = (band_rms >= self.min_band_rms_g and
                    power_share >= self.min_band_power_share)
        label = 'tremor_like_motion' if detected else 'normal_activity'
        return label, peak_frequency, band_rms
