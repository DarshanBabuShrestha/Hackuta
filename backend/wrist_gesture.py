"""Signed gyroscope-peak state machine for experimental wrist confirmation."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class WristGestureConfig:
    rotation_axis: str | None = None
    first_direction: str = 'either'
    rotation_threshold_dps: float = 70.0
    release_threshold_dps: float = 18.0
    integration_deadzone_dps: float = 12.0
    minimum_rotation_degrees: float = 30.0
    minimum_rotation_seconds: float = 0.10
    maximum_rotation_seconds: float = 1.4
    maximum_gap_between_twists_seconds: float = 1.5
    stillness_threshold_dps: float = 18.0
    stillness_seconds: float = 1.0
    sequence_timeout_seconds: float = 5.0
    maximum_sample_gap_seconds: float = 0.15
    smoothing_samples: int = 3
    axis_dominance_ratio: float = 1.0


class WristGestureRecognizer:
    """Find signed gyro lobes on one axis, then require one continuous quiet second.

    Gyroscope values are degrees/second from the Arduino wire protocol. Every
    duration is based on `Sample.timestamp_ms`; host rendering time is unused.
    """

    STAGES = ('WAITING', 'FIRST_TWIST', 'OPPOSITE_TWIST', 'STILLNESS', 'CONFIRMED')
    AXES = ('x', 'y', 'z')

    def __init__(self, config=None):
        self.config = config or WristGestureConfig()
        self._filter = deque(maxlen=max(1, int(self.config.smoothing_samples)))
        self.markers = deque(maxlen=100)
        self.reset()

    def reset(self, reason='Armed; gesture detection starts during CONFIRMING'):
        self.last_timestamp_ms = None
        self._filter.clear()
        self.stage = 'WAITING'
        self.axis = None
        self.first_direction = 0
        self.second_direction = 0
        self.first_angle_deg = 0.0
        self.second_angle_deg = 0.0
        self.first_started_ms = None
        self.second_started_ms = None
        self.first_quiet_since_ms = None
        self.still_since_ms = None
        self.last_rejection_reason = ''
        self.reason = reason
        self.smoothed = (0.0, 0.0, 0.0)
        self.status = {}
        self._publish_status(0)

    def _elapsed_ms(self, later, earlier):
        return (int(later) - int(earlier)) % (2 ** 32)

    def _mark(self, stage, timestamp_ms, direction=0, reason=None):
        marker = {'timestamp_ms': int(timestamp_ms), 'stage': stage,
                  'axis': self.AXES[self.axis] if self.axis is not None else None,
                  'direction': direction}
        if reason:
            marker['reason'] = reason
        self.markers.append(marker)

    def _publish_status(self, timestamp_ms):
        recent = [marker for marker in self.markers
                  if self._elapsed_ms(timestamp_ms, marker['timestamp_ms']) <= 5000]
        self.status = {
            'stage': self.stage,
            'reason': self.reason,
            'rejection_reason': self.last_rejection_reason or None,
            'axis': self.AXES[self.axis] if self.axis is not None else None,
            'first_direction': self.first_direction or None,
            'first_angle_deg': round(self.first_angle_deg, 1),
            'second_angle_deg': round(self.second_angle_deg, 1),
            'stillness_seconds': round(self._elapsed_ms(timestamp_ms, self.still_since_ms) / 1000, 2)
            if self.still_since_ms is not None and self.stage in ('STILLNESS', 'CONFIRMED') else 0.0,
            'markers': recent,
        }

    def _reject(self, reason, timestamp_ms):
        self.last_rejection_reason = reason
        self.reason = reason
        self._mark('REJECTED', timestamp_ms, reason=reason)
        self.stage = 'WAITING'
        self.axis = None
        self.first_direction = 0
        self.second_direction = 0
        self.first_angle_deg = 0.0
        self.second_angle_deg = 0.0
        self.first_started_ms = None
        self.second_started_ms = None
        self.first_quiet_since_ms = None
        self.still_since_ms = None

    def _begin_first(self, axis, direction, timestamp_ms):
        self.last_rejection_reason = ''
        self.axis = axis
        self.first_direction = direction
        self.first_angle_deg = 0.0
        self.first_started_ms = timestamp_ms
        self.first_quiet_since_ms = None
        self.stage = 'FIRST_TWIST'
        self.reason = f'First {"positive" if direction > 0 else "negative"} gyro lobe on {self.AXES[axis].upper()}'
        self._mark('FIRST_TWIST', timestamp_ms, direction)

    def _angle_increment(self, rate_dps, direction, delta_seconds):
        directed_rate = direction * rate_dps
        if directed_rate <= self.config.integration_deadzone_dps:
            return 0.0
        return directed_rate * delta_seconds

    def feed(self, sample, *, confirming, seizure_active=False, live_sensor=True):
        """Return True once when the opposite-peak plus stillness sequence completes."""
        timestamp = int(sample.timestamp_ms) % (2 ** 32)
        if not live_sensor:
            self.reset('Replay data cannot cancel incidents')
            self.last_timestamp_ms = timestamp
            self._publish_status(timestamp)
            return False
        if not confirming:
            self.reset('Armed; gesture detection starts during CONFIRMING')
            self.last_timestamp_ms = timestamp
            self._publish_status(timestamp)
            return False
        if seizure_active:
            self.reset('Blocked while seizure-like movement or an episode is active')
            self.last_timestamp_ms = timestamp
            self._publish_status(timestamp)
            return False

        if self.last_timestamp_ms is None:
            delta_ms = 0
        else:
            delta_ms = self._elapsed_ms(timestamp, self.last_timestamp_ms)
            if delta_ms == 0:
                return False
            if delta_ms > self.config.maximum_sample_gap_seconds * 1000:
                self._filter.clear()
                self._reject('Timeout: serial sample gap interrupted the sequence', timestamp)
        delta_seconds = min(delta_ms / 1000.0, self.config.maximum_sample_gap_seconds)
        self.last_timestamp_ms = timestamp
        raw_gyro = tuple(float(value) for value in sample.values[3:6])
        self._filter.append(raw_gyro)
        self.smoothed = tuple(sum(row[axis] for row in self._filter) / len(self._filter)
                              for axis in range(3))
        magnitudes = tuple(abs(value) for value in self.smoothed)
        vector_magnitude = sum(value * value for value in self.smoothed) ** 0.5
        peak_axis = max(range(3), key=lambda index: magnitudes[index])
        peak_rate = self.smoothed[peak_axis]
        fired = False

        if self.stage == 'CONFIRMED':
            self.reason = 'Wrist gesture confirmed'
        elif self.stage == 'WAITING':
            if magnitudes[peak_axis] >= self.config.rotation_threshold_dps:
                if (self.config.rotation_axis is not None and
                        self.AXES[peak_axis] != self.config.rotation_axis):
                    self.reason = 'Wrong axis: waiting for the configured rotation axis'
                elif magnitudes[peak_axis] < self.config.axis_dominance_ratio * max(
                        magnitudes[i] for i in range(3) if i != peak_axis):
                    self.reason = 'Ambiguous rotation axis; no dominant signed peak'
                else:
                    direction = 1 if peak_rate > 0 else -1
                    self._begin_first(peak_axis, direction, timestamp)
                    self.first_angle_deg += self._angle_increment(
                        peak_rate, direction, delta_seconds)
            else:
                self.reason = 'Waiting for first signed gyro peak'

        elif self.stage == 'FIRST_TWIST':
            elapsed = self._elapsed_ms(timestamp, self.first_started_ms) / 1000.0
            rate = self.smoothed[self.axis]
            signed_rate = rate * self.first_direction
            if elapsed > self.config.maximum_gap_between_twists_seconds:
                self._reject('Timeout: opposite twist did not arrive in time', timestamp)
            elif elapsed > self.config.sequence_timeout_seconds:
                self._reject('Timeout: gesture sequence expired', timestamp)
            elif elapsed > self.config.maximum_rotation_seconds:
                self._reject('Timeout: first rotation lasted too long', timestamp)
            elif signed_rate <= -self.config.rotation_threshold_dps:
                if (elapsed < self.config.minimum_rotation_seconds or
                        self.first_angle_deg < self.config.minimum_rotation_degrees):
                    self._reject('Insufficient angle in the first twist', timestamp)
                else:
                    self.second_direction = -self.first_direction
                    self.second_started_ms = timestamp
                    self.second_angle_deg = self._angle_increment(
                        rate, self.second_direction, delta_seconds)
                    self.stage = 'OPPOSITE_TWIST'
                    self.reason = f'Opposite signed peak detected on {self.AXES[self.axis].upper()}'
                    self._mark('OPPOSITE_TWIST', timestamp, self.second_direction)
            elif signed_rate >= self.config.rotation_threshold_dps:
                if (self.first_quiet_since_ms is not None and
                        self._elapsed_ms(timestamp, self.first_quiet_since_ms) >=
                        self.config.minimum_rotation_seconds * 1000):
                    self._reject('Wrong direction: repeated the first rotation direction', timestamp)
                else:
                    self.first_angle_deg += self._angle_increment(
                        rate, self.first_direction, delta_seconds)
                    self.first_quiet_since_ms = None
            else:
                self.first_angle_deg += self._angle_increment(
                    rate, self.first_direction, delta_seconds)
                if abs(rate) <= self.config.release_threshold_dps:
                    if self.first_quiet_since_ms is None:
                        self.first_quiet_since_ms = timestamp
                else:
                    self.first_quiet_since_ms = None

        elif self.stage == 'OPPOSITE_TWIST':
            elapsed = self._elapsed_ms(timestamp, self.second_started_ms) / 1000.0
            rate = self.smoothed[self.axis]
            signed_rate = rate * self.second_direction
            sequence_elapsed = self._elapsed_ms(timestamp, self.first_started_ms) / 1000.0
            if sequence_elapsed > self.config.sequence_timeout_seconds:
                self._reject('Timeout: gesture sequence expired', timestamp)
            elif signed_rate <= -self.config.rotation_threshold_dps:
                self._reject('Wrong direction: rotation reversed before the second twist completed', timestamp)
            else:
                self.second_angle_deg += self._angle_increment(
                    rate, self.second_direction, delta_seconds)
                if elapsed > self.config.maximum_rotation_seconds:
                    self._reject('Timeout: opposite rotation lasted too long', timestamp)
                elif abs(rate) <= self.config.release_threshold_dps:
                    if (elapsed < self.config.minimum_rotation_seconds or
                            self.second_angle_deg < self.config.minimum_rotation_degrees):
                        self._reject('Insufficient angle in the opposite twist', timestamp)
                    else:
                        self.stage = 'STILLNESS'
                        self.still_since_ms = (timestamp if vector_magnitude <=
                                               self.config.stillness_threshold_dps else None)
                        self.reason = ('Holding still for one continuous second'
                                       if self.still_since_ms is not None else
                                       'Insufficient stillness after the twists')
                        if self.still_since_ms is None:
                            self.last_rejection_reason = self.reason
                        if self.still_since_ms is not None:
                            self._mark('STILLNESS', timestamp, self.second_direction)

        elif self.stage == 'STILLNESS':
            elapsed = self._elapsed_ms(timestamp, self.first_started_ms) / 1000.0
            if elapsed > self.config.sequence_timeout_seconds:
                self._reject('Timeout: one-second stillness was not completed', timestamp)
            elif vector_magnitude <= self.config.stillness_threshold_dps:
                if self.still_since_ms is None:
                    self.still_since_ms = timestamp
                    self._mark('STILLNESS', timestamp, self.second_direction)
                quiet = self._elapsed_ms(timestamp, self.still_since_ms) / 1000.0
                reset_note = ('; Insufficient stillness: previous hold was interrupted'
                              if self.last_rejection_reason else '')
                self.reason = (f'Stillness {quiet:.1f}/{self.config.stillness_seconds:.1f} seconds'
                               f'{reset_note}')
                if quiet >= self.config.stillness_seconds:
                    self.stage = 'CONFIRMED'
                    self.reason = 'Wrist gesture confirmed'
                    self._mark('CONFIRMED', timestamp, self.second_direction)
                    fired = True
            else:
                self.still_since_ms = None
                self.reason = 'Insufficient stillness: gyroscope activity resumed before one second'
                self.last_rejection_reason = self.reason

        self._publish_status(timestamp)
        return fired
