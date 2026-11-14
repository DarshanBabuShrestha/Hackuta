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
from backend.seizure_detector import SeizureMotionDetector
from backend.serial_bridge import SerialBridge, Sample, parse_message


def voice_confirmation(verifier, results):
    try:
        results.put(verifier.confirm_fall())
    except Exception as exc:
        results.put(exc)


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
    parser.add_argument('--voice-confirmation', action='store_true',
                        help='Enroll the owner and use microphone confirmation after a fall')
    parser.add_argument('--voice-model-dir', type=Path)
    parser.add_argument('--seizure-threshold', type=float, default=.5,
                        help='Experimental simulated-seizure-like movement threshold')
    args = parser.parse_args(argv)
    if not args.port and not args.simulate:
        parser.error('Supply --port COM4 or --simulate PATH')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    detector = FallDetector(threshold=args.threshold, cooldown=args.cooldown)
    seizure_detector = SeizureMotionDetector(threshold=args.seizure_threshold)
    incident = IncidentManager()
    voice = None
    voice_results = queue.Queue()
    if args.voice_confirmation:
        import importlib.util
        voice_path = Path(__file__).parents[1] / 'voice-ai' / 'speech.py'
        spec = importlib.util.spec_from_file_location('odyssey_voice', voice_path)
        voice_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(voice_module)
        voice = voice_module.VoiceVerifier(args.voice_model_dir)
        print('=== VOICE REGISTRATION ===', flush=True)
        voice.enroll()
        print('Voice registered for this session.', flush=True)
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
                    continue
        except Exception as exc:
            collection_errors.append(exc)
            logging.exception('Sensor collection stopped')
        finally:
            stop.set()

    threading.Thread(target=collect, daemon=True).start()
    frame_index = 0
    seizure_was_detected = False
    try:
        while True:
            while not commands.empty():
                if commands.get_nowait() in ('c', 'cancel', 'okay'):
                    incident.cancel()
            incident.tick()
            while not voice_results.empty():
                result = voice_results.get_nowait()
                if isinstance(result, Exception):
                    print(f'Voice confirmation failed: {result}', flush=True)
                else:
                    score, is_owner, said_okay, spoken_text = result
                    print(f'Voice response: {spoken_text!r}; similarity: {score:.3f}', flush=True)
                    if is_owner and said_okay and incident.cancel():
                        print('Owner voice tentatively verified; incident cancelled.', flush=True)
                    else:
                        print('Voice confirmation not verified; alert remains active.', flush=True)
            try:
                window = windows.get(timeout=.1)
            except queue.Empty:
                if stop.is_set() and windows.empty() and incident.state != 'CONFIRMING':
                    break
                continue
            # SerialBridge emits 32 samples with a 16-sample hop at 50 Hz.
            # Replay windows use the same nominal timing.
            end_ms = 620 + frame_index * 320
            frame_index += 1
            seizure_result = seizure_detector.push_window(window, end_ms)
            if seizure_result is not None:
                seizure_label, _ = seizure_result
                seizure_detected = seizure_label == 'simulated_seizure_like_motion'
                if seizure_detected and not seizure_was_detected:
                    print('SIMULATED SEIZURE-LIKE MOVEMENT DETECTED', flush=True)
                seizure_was_detected = seizure_detected
            _, _, trigger = detector.predict(window)
            if trigger:
                if incident.candidate() and voice:
                    threading.Thread(target=voice_confirmation,
                                     args=(voice, voice_results), daemon=True).start()
            else:
                print(f'Status: {incident.state}', flush=True)
                incident.candidate()
    except KeyboardInterrupt:
        print('Stopping.')
    if collection_errors:
        raise RuntimeError('Sensor collection failed') from collection_errors[0]


if __name__ == '__main__':
    main()
