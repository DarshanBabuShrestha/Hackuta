"""Run live USB or BITS-2 replay. Press c then Enter to cancel an incident."""
import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from backend.fall_detector import FallDetector
from backend.features import G, WINDOW_SIZE
from backend.incident_manager import IncidentManager
from backend.serial_bridge import SerialBridge, Sample, parse_message


def keyboard_reader(commands):
    for line in sys.stdin:
        commands.put(line.strip().lower())


def replay(path, bridge, speed=1.0):
    """Replay BITS-2 CSV (acg/gyro) or Arduino IMU CSV as timed windows."""
    if path.suffix.lower() == '.csv' and 'Dataset' in path.parts:
        from ml.train_odyssey_windows import load_recording
        data = load_recording(path)
        if data is None:
            raise ValueError('Recording has too few acg/gyro samples')
        wire = data.copy()
        wire[:, :3] /= G
        wire[:, 3:] *= 180 / np.pi
        lines = [f"IMU,{i * 20},{','.join(map(str, row))}" for i, row in enumerate(wire)]
    else:
        lines = path.read_text().splitlines()
    for line in lines:
        try:
            window = bridge.push(parse_message(line))
        except (ValueError, UnicodeError) as exc:
            logging.warning('Skipping malformed replay message: %s', exc)
            continue
        if window is not None:
            yield window
        if speed > 0:
            time.sleep(.02 / speed)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experimental Odyssey Guardian fall detector')
    parser.add_argument('--port', help='Arduino COM port, e.g. COM4')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--simulate', type=Path, help='BITS-2 recording or Arduino IMU CSV')
    parser.add_argument('--speed', type=float, default=1, help='Replay speed; 0 for fastest')
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--cooldown', type=float, default=20)
    args = parser.parse_args(argv)
    if not args.port and not args.simulate:
        parser.error('Supply --port COM4 or --simulate PATH')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    detector = FallDetector(threshold=args.threshold, cooldown=args.cooldown)
    incident = IncidentManager()
    commands = queue.Queue()
    threading.Thread(target=keyboard_reader, args=(commands,), daemon=True).start()
    bridge = SerialBridge(args.port or 'simulation', baud=args.baud)
    source = replay(args.simulate, bridge, args.speed) if args.simulate else bridge.windows()
    # Sensor collection runs in a worker so the countdown continues even if
    # serial.readline() waits or a replay ends.
    windows = queue.Queue(maxsize=8)
    stop = threading.Event()
    collection_errors = []

    def collect():
        try:
            for window in source:
                try:
                    windows.put_nowait(window)
                except queue.Full:
                    logging.warning('Processing behind; dropping sensor window')
        except Exception as exc:
            collection_errors.append(exc)
            logging.exception('Sensor collection stopped')
        finally:
            stop.set()

    threading.Thread(target=collect, daemon=True).start()
    try:
        while True:
            while not commands.empty():
                if commands.get_nowait() in ('c', 'cancel', 'okay'):
                    incident.cancel()
            incident.tick()
            try:
                window = windows.get(timeout=.1)
            except queue.Empty:
                if stop.is_set() and windows.empty() and incident.state != 'CONFIRMING':
                    break
                continue
            print('Sensor window collected', flush=True)
            label, probability, trigger = detector.predict(window)
            print(f'Fall probability: {probability:.2f}', flush=True)
            if trigger:
                incident.candidate()
            else:
                print(f'Status: {incident.state}', flush=True)
    except KeyboardInterrupt:
        print('Stopping.')
    if collection_errors:
        raise RuntimeError('Sensor collection failed') from collection_errors[0]


if __name__ == '__main__':
    main()
