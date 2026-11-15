"""Run live USB or BITS-2 replay. Press c then Enter to cancel an incident."""
import argparse
import json
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
from backend.tremor_detector import TremorMotionDetector


def voice_confirmation(verifier, results, incident):
    try:
        results.put(verifier.confirm_fall(
            on_started=incident.start_response_window,
            on_recorded=incident.response_captured,
        ))
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
    parser.add_argument('--confirmation-seconds', type=float, default=15,
                        help='Seconds to wait for wearer confirmation after a possible fall')
    parser.add_argument('--voice-confirmation', action='store_true',
                        help='Enroll the owner and use microphone confirmation after a fall')
    parser.add_argument('--voice-model-dir', type=Path)
    parser.add_argument('--seizure-threshold', type=float, default=.5,
                        help='Experimental simulated-seizure-like movement threshold')
    parser.add_argument('--mode', choices=('all', 'fall', 'tremor', 'seizure', 'epilepsy'), default='all',
                        help='Run all detectors or one detector for an isolated test')
    args = parser.parse_args(argv)
    if not args.port and not args.simulate:
        parser.error('Supply --port COM4 or --simulate PATH')
    if args.voice_confirmation and args.mode not in ('all', 'fall'):
        parser.error('--voice-confirmation requires --mode all or --mode fall')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_fall = args.mode in ('all', 'fall')
    run_tremor = args.mode in ('all', 'tremor')
    run_seizure = args.mode in ('all', 'seizure', 'epilepsy')
    detector = FallDetector(threshold=args.threshold, cooldown=args.cooldown) if run_fall else None
    tremor_detector = TremorMotionDetector() if run_tremor else None
    seizure_detector = SeizureMotionDetector(threshold=args.seizure_threshold) if run_seizure else None
    incident = IncidentManager(duration=4 if args.voice_confirmation else args.confirmation_seconds)
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
    tremor_was_detected = False
    try:
        while True:
            while not commands.empty():
                command = commands.get_nowait()
                if command in ('c', 'cancel', 'okay'):
                    incident.cancel()
                elif command in ('help', 'need help'):
                    if incident.state in ('CONFIRMING', 'VERIFYING'):
                        print('Wearer requested help.', flush=True)
                        incident.caregiver_pending()
            incident.tick()
            while not voice_results.empty():
                result = voice_results.get_nowait()
                if isinstance(result, Exception):
                    print(f'Voice confirmation failed: {result}', flush=True)
                    incident.caregiver_pending()
                else:
                    score, is_owner, said_okay, spoken_text = result
                    print(f'Voice response: {spoken_text!r}; similarity: {score:.3f}', flush=True)
                    if is_owner and said_okay and incident.cancel():
                        print('Owner voice verified; status: MONITORING', flush=True)
                    else:
                        print('Voice confirmation not verified; alert remains active.', flush=True)
                        incident.caregiver_pending()
            try:
                window = windows.get(timeout=.1)
            except queue.Empty:
                if stop.is_set() and windows.empty() and incident.state not in ('CONFIRMING', 'VERIFYING'):
                    break
                continue
            # SerialBridge emits 32 samples with a 16-sample hop at 50 Hz.
            # Replay windows use the same nominal timing.
            end_ms = 620 + frame_index * 320
            frame_index += 1
            print('Sensor window collected', flush=True)
            axis_rms = np.std(np.asarray(window, dtype=float)[:, :3], axis=0)
            motion_point = {
                'x_g_rms': round(float(axis_rms[0]), 4),
                'y_g_rms': round(float(axis_rms[1]), 4),
                'z_g_rms': round(float(axis_rms[2]), 4),
            }
            print(f'Motion window: {json.dumps(motion_point)}', flush=True)
            if seizure_detector:
                seizure_result = seizure_detector.push_window(window, end_ms)
                if seizure_result is not None:
                    seizure_label, seizure_probability = seizure_result
                    seizure_detected = seizure_label == 'simulated_seizure_like_motion'
                    print(f'Seizure-like motion: {seizure_label}; probability: '
                          f'{seizure_probability:.2f}', flush=True)
                    if seizure_detected and not seizure_was_detected:
                        print('SIMULATED SEIZURE-LIKE MOVEMENT DETECTED', flush=True)
                    seizure_was_detected = seizure_detected
            if tremor_detector:
                tremor_result = tremor_detector.push_window(window)
                if tremor_result is not None:
                    tremor_label, peak_hz, band_rms_g = tremor_result
                    tremor_detected = tremor_label == 'tremor_like_motion'
                    print(f'Tremor-like motion: {tremor_label}; dominant frequency: '
                          f'{peak_hz:.1f} Hz; band RMS: {band_rms_g:.3f} g', flush=True)
                    if tremor_detected and not tremor_was_detected:
                        print('TREMOR-LIKE MOTION DETECTED', flush=True)
                    tremor_was_detected = tremor_detected
            if detector:
                _, _, trigger = detector.predict(window)
                print(f'Status: {incident.state}', flush=True)
                if trigger and incident.state == 'MONITORING':
                    prompt = 'Starting 4-second voice confirmation.' if voice else 'Are you okay? Type c then Enter to cancel.'
                    if incident.candidate(prompt=prompt, defer_timer=bool(voice)) and voice:
                        threading.Thread(target=voice_confirmation,
                                         args=(voice, voice_results, incident), daemon=True).start()
    except KeyboardInterrupt:
        print('Stopping.')
    if collection_errors:
        raise RuntimeError('Sensor collection failed') from collection_errors[0]


if __name__ == '__main__':
    main()
