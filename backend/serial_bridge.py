"""Arduino CSV protocol and reconnecting serial reader."""
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

from .features import WINDOW_SIZE

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    timestamp_ms: int
    values: tuple[float, float, float, float, float, float]


def parse_message(line):
    """Parse `IMU,millis,ax,ay,az,gx,gy,gz`; ignore header lines."""
    if isinstance(line, bytes):
        line = line.decode('ascii', errors='strict')
    parts = line.strip().split(',')
    if len(parts) != 8 or parts[0] != 'IMU':
        raise ValueError('Expected IMU,t_ms,ax,ay,az,gx,gy,gz')
    timestamp = int(parts[1])
    values = tuple(float(part) for part in parts[2:])
    if timestamp < 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid timestamp or non-finite sensor value')
    if any(abs(v) > 32 for v in values[:3]) or any(abs(v) > 4000 for v in values[3:]):
        raise ValueError('Sensor value outside plausible IMU range')
    return Sample(timestamp, values)


class SerialBridge:
    def __init__(self, port, baud=115200, window_size=WINDOW_SIZE, hop=16):
        self.port, self.baud, self.hop = port, baud, hop
        self.buffer = deque(maxlen=window_size)
        self.count = 0
        self.last_timestamp = None

    def push(self, sample):
        # millis() wraps after about 49 days; a reboot or gap clears the window.
        if self.last_timestamp is not None:
            delta = (sample.timestamp_ms - self.last_timestamp) % (2 ** 32)
            if delta == 0 or delta > 200:
                self.buffer.clear()
                self.count = 0
                LOG.warning('Sensor timing gap; starting a new window')
        self.last_timestamp = sample.timestamp_ms
        self.buffer.append(sample.values)
        self.count += 1
        if len(self.buffer) == self.buffer.maxlen and (self.count - self.buffer.maxlen) % self.hop == 0:
            return list(self.buffer)
        return None

    def windows(self):
        import serial
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as device:
                    LOG.info('Arduino connected: %s', self.port)
                    LOG.info('IMU streaming...')
                    self.buffer.clear()
                    self.count = 0
                    self.last_timestamp = None
                    while True:
                        raw = device.readline()
                        if not raw:
                            continue
                        try:
                            window = self.push(parse_message(raw))
                            if window is not None:
                                yield window
                        except (ValueError, UnicodeError) as exc:
                            LOG.warning('Skipping malformed message: %s', exc)
            except (serial.SerialException, OSError) as exc:
                LOG.warning('Serial disconnected: %s; retrying in 2 seconds', exc)
                time.sleep(2)
