"""Run live USB or BITS-2 replay. Press c then Enter to cancel an incident."""
import argparse
import json
import logging
import queue
import socket
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


def voice_confirmation(verifier, results, incident, dashboard=None):
    def started():
        if incident.start_response_window() and dashboard:
            dashboard.publish(incident=incident.state,
                              voice_status="Listening for 4 seconds: say 'I'm okay' now")

    def recorded():
        if incident.response_captured() and dashboard:
            dashboard.publish(incident=incident.state, voice_status='Verifying voice response')

    try:
        results.put(verifier.confirm_fall(
            on_started=started,
            on_recorded=recorded,
        ))
    except Exception as exc:
        results.put(exc)


def load_voice_model(model_dir, events):
    try:
        import importlib.util
        voice_path = Path(__file__).parents[1] / 'voice-ai' / 'speech.py'
        spec = importlib.util.spec_from_file_location('odyssey_voice', voice_path)
        voice_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(voice_module)
        events.put(('loaded', voice_module.VoiceVerifier(model_dir), voice_module.DURATION))
    except Exception as exc:
        events.put(('load_error', exc, None))


def enroll_voice(verifier, events, dashboard):
    try:
        verifier.enroll(
            wait_for_enter=False,
            on_started=lambda: dashboard.publish(voice_status='Recording enrollment for 4 seconds'),
            on_recorded=lambda: dashboard.publish(voice_status='Saving voice enrollment'),
        )
        events.put(('enrolled', None, None))
    except Exception as exc:
        events.put(('enroll_error', exc, None))


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
    parser.add_argument('--port', help='Arduino COM port, e.g. COM9')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--simulate', type=Path, help='BITS-2 recording or Arduino IMU CSV')
    parser.add_argument('--speed', type=float, default=1, help='Replay speed; 0 for fastest')
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--cooldown', type=float, default=20)
    parser.add_argument('--confirmation-seconds', type=float, default=15,
                        help='Seconds to wait for wearer confirmation after a possible fall')
    parser.add_argument('--voice-confirmation', action='store_true',
                        help='Enable the dashboard with voice confirmation (also enabled by --web)')
    parser.add_argument('--voice-model-dir', type=Path)
    parser.add_argument('--seizure-threshold', type=float, default=.5,
                        help='Experimental simulated-seizure-like movement threshold')
    parser.add_argument('--mode', choices=('all', 'fall', 'tremor', 'seizure', 'epilepsy'), default='all',
                        help='Run all detectors or one detector for an isolated test')
    parser.add_argument('--web', action='store_true', help='Serve the local live dashboard')
    parser.add_argument('--no-web', action='store_true', help='Run without the local dashboard')
    parser.add_argument('--web-port', type=int, default=8000, help='Local dashboard port')
    args = parser.parse_args(argv)
    if args.no_web and args.voice_confirmation:
        parser.error('--voice-confirmation requires the dashboard')
    args.web = (args.web or bool(args.port and not args.simulate) or
                args.voice_confirmation) and not args.no_web
    if not args.port and not args.simulate:
        parser.error('Supply --port COM9 or --simulate PATH')
    if args.voice_confirmation and args.mode == 'tremor':
        parser.error('--voice-confirmation requires fall or seizure mode')
    if args.web:
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', args.web_port))
        except OSError as exc:
            parser.error(f'Dashboard port {args.web_port} is in use; stop the other backend or choose --web-port: {exc}')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_fall = args.mode in ('all', 'fall')
    run_tremor = args.mode in ('all', 'tremor')
    run_seizure = args.mode in ('all', 'seizure', 'epilepsy')
    voice_enabled = args.web and (run_fall or run_seizure)
    detector = FallDetector(threshold=args.threshold, cooldown=args.cooldown) if run_fall else None
    tremor_detector = TremorMotionDetector() if run_tremor else None
    seizure_detector = SeizureMotionDetector(threshold=args.seizure_threshold) if run_seizure else None
    commands = queue.Queue()
    dashboard = None
    if args.web:
        from backend.dashboard import Dashboard, serve
        dashboard = Dashboard(commands, str(args.simulate or args.port),
                              enabled={'fall': run_fall, 'tremor': run_tremor,
                                       'seizure': run_seizure},
                              voice_enabled=voice_enabled)

    def incident_message(message):
        print(message, flush=True)
        if dashboard:
            dashboard.event(message)
            dashboard.publish(incident=incident.state, incident_type=incident.kind)
            if voice_enabled and voice_recording and message == 'CAREGIVER ALERT REQUIRED':
                dashboard.publish(voice_status='Voice confirmation did not complete')

    incident = IncidentManager(duration=args.confirmation_seconds, emit=incident_message)
    if dashboard:
        threading.Thread(target=serve, args=(dashboard, args.web_port), daemon=True).start()
        print(f'Dashboard: http://127.0.0.1:{args.web_port}', flush=True)
    voice = None
    voice_duration = 4
    enrolling = False
    voice_recording = False
    voice_events = queue.Queue()
    voice_results = queue.Queue()
    threading.Thread(target=keyboard_reader, args=(commands,), daemon=True).start()
    bridge = SerialBridge(args.port or 'simulation', baud=args.baud,
                          on_status=(lambda status: dashboard.publish(sensor_status=status))
                          if dashboard else None)
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
    if voice_enabled:
        threading.Thread(target=load_voice_model,
                         args=(args.voice_model_dir, voice_events), daemon=True).start()
    frame_index = 0
    seizure_was_detected = False
    tremor_was_detected = False
    try:
        while True:
            while not voice_events.empty():
                kind, value, duration = voice_events.get_nowait()
                if kind == 'loaded':
                    voice = value
                    voice_duration = duration
                    dashboard.publish(voice_status='Ready to enroll',
                                      voice_device=voice.input_device_name,
                                      voice_can_enroll=True)
                elif kind == 'load_error':
                    logging.error('Voice model unavailable: %s', value)
                    dashboard.publish(voice_status=f'Voice model unavailable: {value}')
                elif kind == 'enrolled':
                    enrolling = False
                    print('Voice registered for this session.', flush=True)
                    dashboard.publish(voice_status='Voice enrolled; ready for falls',
                                      voice_enrolled=True, voice_can_enroll=True)
                elif kind == 'enroll_error':
                    enrolling = False
                    logging.error('Voice enrollment failed: %s', value)
                    dashboard.publish(voice_status=f'Enrollment failed: {value}',
                                      voice_can_enroll=True)
            previous_state = incident.state
            while not commands.empty():
                command = commands.get_nowait()
                if command in ('c', 'cancel', 'okay'):
                    if incident.cancel() and dashboard and voice:
                        dashboard.publish(voice_status='Cancelled manually')
                elif command in ('help', 'need help'):
                    if incident.state in ('CONFIRMING', 'VERIFYING'):
                        print('Wearer requested help.', flush=True)
                        incident.caregiver_pending()
                elif command == 'enroll_voice' and voice and not enrolling and not voice_recording and incident.state == 'MONITORING':
                    enrolling = True
                    dashboard.publish(voice_status='Starting enrollment', voice_can_enroll=False)
                    threading.Thread(target=enroll_voice,
                                     args=(voice, voice_events, dashboard), daemon=True).start()
            incident.tick()
            if dashboard and incident.state != previous_state:
                dashboard.publish(incident=incident.state, incident_type=incident.kind)
            while not voice_results.empty():
                result = voice_results.get_nowait()
                voice_recording = False
                if dashboard and not enrolling:
                    dashboard.publish(voice_can_enroll=True)
                if incident.state not in ('CONFIRMING', 'VERIFYING'):
                    continue
                if isinstance(result, Exception):
                    print(f'Voice confirmation failed: {result}', flush=True)
                    incident.caregiver_pending()
                    if dashboard:
                        dashboard.publish(voice_status=f'Voice check failed: {result}')
                else:
                    score, is_owner, said_okay, spoken_text, recognition_error = result
                    print(f'Voice response: {spoken_text!r}; similarity: {score:.3f}; '
                          f'microphone level: {voice.last_level:.4f}', flush=True)
                    if is_owner and said_okay and incident.cancel():
                        print('Owner voice verified; status: MONITORING', flush=True)
                        if dashboard:
                            dashboard.publish(voice_status='Owner voice confirmed')
                    else:
                        print('Voice confirmation not verified; alert remains active.', flush=True)
                        incident.caregiver_pending()
                        if dashboard:
                            if recognition_error:
                                reason = recognition_error
                            elif not said_okay:
                                reason = f'Heard {spoken_text!r}; say "I\'m okay"'
                            else:
                                reason = 'Voice did not match enrolled speaker'
                            dashboard.publish(voice_status=reason)
                if dashboard:
                    dashboard.publish(incident=incident.state)
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
            update = {'motion': motion_point, 'windows': frame_index}
            seizure_trigger = False
            fall_trigger = False
            if seizure_detector:
                seizure_result = seizure_detector.push_window(window, end_ms)
                if seizure_result is not None:
                    seizure_label, seizure_probability = seizure_result
                    seizure_detected = seizure_label == 'simulated_seizure_like_motion'
                    print(f'Seizure-like motion: {seizure_label}; probability: '
                          f'{seizure_probability:.2f}', flush=True)
                    if seizure_detected and not seizure_was_detected:
                        seizure_trigger = True
                        if dashboard and incident.state != 'MONITORING':
                            dashboard.event('Seizure-like motion detected (experimental)')
                    seizure_was_detected = seizure_detected
                    update['seizure'] = {'label': seizure_label, 'probability': seizure_probability}
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
                    update['tremor'] = {'label': tremor_label, 'peak_hz': peak_hz,
                                        'band_rms_g': band_rms_g}
            if detector:
                _, _, fall_trigger = detector.predict(window)
                print(f'Status: {incident.state}', flush=True)
            candidate_kind = ('seizure_like_motion' if seizure_trigger else
                              'fall' if fall_trigger else None)
            if candidate_kind and incident.state == 'MONITORING':
                voice_active = bool(voice and voice.owner_voice is not None and not enrolling)
                incident.duration = voice_duration if voice_active else args.confirmation_seconds
                prompt = 'Starting voice confirmation.' if voice_active else 'Are you okay? Type c then Enter to cancel.'
                if incident.candidate(prompt=prompt, defer_timer=voice_active,
                                      kind=candidate_kind) and voice_active:
                    if dashboard:
                        dashboard.publish(voice_status='Starting voice confirmation',
                                          voice_can_enroll=False)
                    voice_recording = True
                    threading.Thread(target=voice_confirmation,
                                     args=(voice, voice_results, incident, dashboard), daemon=True).start()
            if dashboard:
                dashboard.publish(incident=incident.state, incident_type=incident.kind, **update)
    except KeyboardInterrupt:
        print('Stopping.')
    if collection_errors:
        raise RuntimeError('Sensor collection failed') from collection_errors[0]


if __name__ == '__main__':
    main()
