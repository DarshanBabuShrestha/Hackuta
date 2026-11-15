"""Run live USB or BITS-2 replay. Press c then Enter to cancel an incident."""
import argparse
import json
import logging
import os
import queue
import re
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from backend.fall_detector import FallDetector
from backend.caregiver_alerts import CaregiverAlerts
from backend.episode_tracker import EpisodeTracker
from backend.features import G, WINDOW_SIZE
from backend.incident_manager import CONFIRMATION_PROMPT, IncidentManager
from backend.seizure_detector import (DEFAULT_THRESHOLD, HIGH_HZ, LOW_HZ,
                                      MIN_BAND_POWER_SHARE, MIN_MOTION_RMS_G, MIN_PEAK_POWER_SHARE,
                                      RHYTHMIC_SECONDS, WINDOW_SIZE as SEIZURE_WINDOW_SIZE,
                                      SeizureMotionDetector)
from backend.serial_bridge import SerialBridge, Sample, parse_message
from backend.speech_output import speak, stop_speaking
from backend.tremor_detector import TremorMotionDetector


VOICE_REJECTED_MESSAGE = 'Voice not recognized. Alerting the caregiver.'
BYSTANDER_ASSISTANCE_MESSAGE = """Hello! I may be unconscious or experiencing a seizure, and I may not be able to ask for help.

If you see me shaking or having a seizure:

Please stay with me and keep me safe.

Move any dangerous objects away from me.

Cushion my head and time the seizure.

Do not hold me down or put anything in my mouth.

When it is safe, gently turn me onto my side.

Call 911 immediately if I am having trouble breathing, I am injured, the seizure lasts 5 minutes or longer, or another seizure starts before I recover. If you are unsure whether this is my first seizure, call for emergency help.

My caregiver has been notified.

Please stay with me until I recover or help arrives."""
_broadcast_can_play = threading.Event()
_broadcast_can_play.set()
_broadcast_active = threading.Event()


def broadcast_first_aid_message(dashboard=None):
    _broadcast_can_play.set()
    _broadcast_active.set()
    if dashboard:
        dashboard.publish(broadcast_active=True, broadcast_paused=False)
    try:
        sentences = re.split(r'(?<=[.!?])\s+', BYSTANDER_ASSISTANCE_MESSAGE.strip())
        for sentence in sentences:
            while True:
                _broadcast_can_play.wait()
                if speak(sentence):
                    break
                if _broadcast_can_play.is_set():
                    return
    finally:
        _broadcast_active.clear()
        _broadcast_can_play.set()
        if dashboard:
            dashboard.publish(broadcast_active=False, broadcast_paused=False)


def toggle_first_aid_broadcast(dashboard=None):
    if not _broadcast_active.is_set():
        return False
    if _broadcast_can_play.is_set():
        _broadcast_can_play.clear()
        stop_speaking()
        paused = True
    else:
        _broadcast_can_play.set()
        paused = False
    if dashboard:
        dashboard.publish(broadcast_active=True, broadcast_paused=paused)
        dashboard.event('Emergency broadcast paused by wearer' if paused
                        else 'Emergency broadcast resumed by wearer')
    return True


def reject_voice_confirmation(incident, dashboard=None):
    print(VOICE_REJECTED_MESSAGE, flush=True)
    speak(VOICE_REJECTED_MESSAGE)
    incident.caregiver_pending()
    if dashboard:
        dashboard.publish(voice_status=VOICE_REJECTED_MESSAGE)


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
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Fall probability threshold')
    parser.add_argument('--cooldown', type=float, default=10,
                        help='Minimum seconds between fall triggers')
    parser.add_argument('--confirmation-seconds', type=float, default=15,
                        help='Seconds to wait for wearer confirmation after a possible fall')
    parser.add_argument('--voice-confirmation', action='store_true',
                        help='Enable the dashboard with voice confirmation (also enabled by --web)')
    parser.add_argument('--voice-model-dir', type=Path)
    parser.add_argument('--seizure-motion-rms-g', type=float, default=MIN_MOTION_RMS_G,
                        help='Experimental minimum one-second dynamic acceleration RMS in g')
    parser.add_argument('--seizure-peak-power-share', type=float, default=MIN_PEAK_POWER_SHARE,
                        help='Experimental minimum FFT peak share of non-DC power')
    parser.add_argument('--seizure-band-power-share', type=float, default=MIN_BAND_POWER_SHARE,
                        help='Experimental minimum rhythmic-band share of non-DC power')
    parser.add_argument('--seizure-low-hz', type=float, default=LOW_HZ)
    parser.add_argument('--seizure-high-hz', type=float, default=HIGH_HZ)
    parser.add_argument('--seizure-rhythmic-seconds', type=float, default=RHYTHMIC_SECONDS)
    parser.add_argument('--seizure-threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--episode-pre-event-seconds', type=float, default=5.0,
                        help='Seconds of timestamped sensor data to retain before an episode onset')
    parser.add_argument('--episode-recovery-seconds', type=float, default=5.0,
                        help='Continuous normal rhythmic-motion recovery required to finish a report')
    parser.add_argument('--episode-burst-min-seconds', type=float, default=0.2,
                        help='Minimum above-average movement interval counted as a burst')
    parser.add_argument('--episode-burst-hysteresis', type=float, default=0.1,
                        help='Relative hysteresis below the average for closing an above-average burst')
    parser.add_argument('--mode', choices=('all', 'fall', 'tremor', 'seizure', 'epilepsy'), default='all',
                        help='Run all detectors or one detector for an isolated test')
    parser.add_argument('--web', action='store_true', help='Serve the local live dashboard')
    parser.add_argument('--no-web', action='store_true', help='Run without the local dashboard')
    parser.add_argument('--web-port', type=int, default=8000, help='Local dashboard port')
    parser.add_argument('--web-host', default='127.0.0.1',
                        help='Bind address; use 0.0.0.0 for a caregiver phone on the same network')
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
                probe.bind((args.web_host, args.web_port))
        except OSError as exc:
            parser.error(f'Dashboard port {args.web_port} is in use; stop the other backend or choose --web-port: {exc}')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_fall = args.mode in ('all', 'fall')
    run_tremor = args.mode in ('all', 'tremor')
    run_seizure = args.mode in ('all', 'seizure', 'epilepsy')
    voice_enabled = args.web and (run_fall or run_seizure)
    detector = (FallDetector(threshold=args.threshold, cooldown=args.cooldown)
                if run_fall else None)
    tremor_detector = TremorMotionDetector() if run_tremor else None
    seizure_detector = (SeizureMotionDetector(
        motion_rms_g=args.seizure_motion_rms_g,
        peak_power_share=args.seizure_peak_power_share,
        band_power_share=args.seizure_band_power_share,
        low_hz=args.seizure_low_hz, high_hz=args.seizure_high_hz,
        rhythmic_seconds=args.seizure_rhythmic_seconds,
        threshold=args.seizure_threshold) if run_seizure else None)
    episode_tracker = (EpisodeTracker(
        pre_event_seconds=args.episode_pre_event_seconds,
        trigger_lookback_seconds=max(args.seizure_rhythmic_seconds,
                                     SEIZURE_WINDOW_SIZE / 16),
        recovery_seconds=args.episode_recovery_seconds,
        burst_min_seconds=args.episode_burst_min_seconds,
        burst_hysteresis=args.episode_burst_hysteresis,
        minimum_motion_rms_g=args.seizure_motion_rms_g) if run_seizure else None)
    commands = queue.Queue()
    dashboard = None
    caregiver_key = os.environ.get('ODYSSEY_CAREGIVER_KEY') or secrets.token_urlsafe(16)
    if args.web:
        from backend.dashboard import Dashboard, serve
        dashboard = Dashboard(commands, str(args.simulate or args.port),
                              enabled={'fall': run_fall, 'tremor': run_tremor,
                                       'seizure': run_seizure},
                              voice_enabled=voice_enabled, caregiver_key=caregiver_key,
                              simulation_waiting=bool(args.simulate),
                              episode_tracker=episode_tracker)

    def alert_message(message):
        print(message, flush=True)
        if dashboard:
            dashboard.event(message)
            dashboard.publish(caregiver_alert=alerts.alert.copy() if alerts.alert else None)

    alerts = CaregiverAlerts(emit=alert_message)

    def incident_message(message):
        print(message, flush=True)
        if dashboard:
            dashboard.event(message)
            dashboard.publish(incident=incident.state, incident_type=incident.kind)
            if voice_enabled and voice_recording and message == 'CAREGIVER ALERT REQUIRED':
                dashboard.publish(voice_status='Voice confirmation did not complete')
        if message == 'CAREGIVER ALERT REQUIRED':
            if alerts.start(incident.kind):
                threading.Thread(target=broadcast_first_aid_message,
                                 args=(dashboard,), daemon=True).start()

    incident = IncidentManager(duration=args.confirmation_seconds, emit=incident_message)
    if dashboard:
        threading.Thread(target=serve, args=(dashboard, args.web_port, args.web_host), daemon=True).start()
        print(f'Dashboard: http://127.0.0.1:{args.web_port}', flush=True)
        print(f'Caregiver page: http://<this-computer-LAN-IP>:{args.web_port}/caregiver', flush=True)
        print(f'Caregiver access key: {caregiver_key}', flush=True)
    voice = None
    voice_duration = 4
    enrolling = False
    voice_recording = False
    voice_events = queue.Queue()
    voice_results = queue.Queue()
    threading.Thread(target=keyboard_reader, args=(commands,), daemon=True).start()
    def sensor_status(status):
        if episode_tracker:
            lowered = status.lower()
            if 'disconnect' in lowered or 'no serial data' in lowered or 'initialization failed' in lowered:
                episode_tracker.mark_disconnected(status)
            elif 'connected' in lowered or 'receiving imu' in lowered:
                episode_tracker.mark_reconnected()
        if dashboard:
            changes = {'sensor_status': status}
            if episode_tracker:
                changes['episode_report'] = episode_tracker.summary()
                changes['episode_history'] = episode_tracker.history_summaries()
            dashboard.publish(**changes)

    bridge = SerialBridge(args.port or 'simulation', baud=args.baud,
                          on_status=sensor_status if dashboard or episode_tracker else None)
    source = replay(args.simulate, bridge, args.speed) if args.simulate else bridge.windows()
    # Sensor collection runs in a worker so the countdown continues even if
    # serial.readline() waits or a replay ends.
    windows = queue.Queue(maxsize=8)
    stop = threading.Event()
    simulation_start = threading.Event()
    if not args.simulate or not args.web:
        simulation_start.set()
    collection_errors = []

    def collect():
        try:
            simulation_start.wait()
            for window in source:
                timestamped_window = list(bridge.latest_samples)[-32:]
                try:
                    windows.put_nowait((window, timestamped_window))
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
    seizure_rhythm_candidate_start_ms = None
    seizure_rhythm_lost_ms = None
    fall_was_detected = False
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
                if isinstance(command, dict) and command.get('command') == 'acknowledge':
                    alerts.acknowledge(command.get('alert_id'))
                    continue
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
                elif command == 'start_simulation' and args.simulate and args.web and not simulation_start.is_set():
                    simulation_start.set()
                    dashboard.publish(simulation_waiting=False)
                    dashboard.event('Sensor replay started')
                elif command == 'pause_broadcast':
                    toggle_first_aid_broadcast(dashboard)
            incident.tick()
            alerts.tick()
            if (incident.state == 'CAREGIVER_PENDING' and alerts.alert and
                    alerts.alert['status'] in ('ACKNOWLEDGED', 'CALL_QUEUED', 'CALL_FAILED')):
                incident.close_caregiver()
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
                    logging.error('Voice confirmation failed: %s', result)
                    reject_voice_confirmation(incident, dashboard)
                else:
                    score, is_owner, said_okay, spoken_text, _recognition_error = result
                    print(f'Voice response: {spoken_text!r}; similarity: {score:.3f}; '
                          f'microphone level: {voice.last_level:.4f}', flush=True)
                    if is_owner and said_okay and incident.cancel():
                        print('Owner voice verified; status: MONITORING', flush=True)
                        if dashboard:
                            dashboard.publish(voice_status='Owner voice confirmed')
                    else:
                        reject_voice_confirmation(incident, dashboard)
                if dashboard:
                    dashboard.publish(incident=incident.state)
            try:
                queued_window = windows.get(timeout=.1)
            except queue.Empty:
                if (stop.is_set() and windows.empty() and
                        (collection_errors or
                         (not args.web and incident.state not in ('CONFIRMING', 'VERIFYING')
                          and not alerts.unfinished))):
                    break
                continue
            # SerialBridge emits 32 samples with a 16-sample hop at 50 Hz.
            # Replay windows use the same nominal timing.
            if (isinstance(queued_window, tuple) and len(queued_window) == 2 and
                    isinstance(queued_window[1], list)):
                window, sensor_samples = queued_window
            else:
                window, sensor_samples = queued_window, []
            end_ms = (sensor_samples[-1].timestamp_ms if sensor_samples
                      else 620 + frame_index * 320)
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
            if len(sensor_samples) >= 2:
                window_axes = np.asarray([sample.values[:3] for sample in sensor_samples], dtype=float)
                centered_window = window_axes - window_axes.mean(axis=0)
                window_motion_rms = float(np.sqrt(np.mean(np.sum(centered_window ** 2, axis=1))))
            else:
                window_motion_rms = None
            if dashboard and bridge.latest_samples:
                latest_ms = bridge.latest_samples[-1].timestamp_ms
                raw = np.asarray([sample.values[:3] for sample in bridge.latest_samples], dtype=float)
                # Per-axis baseline removal centers gravity at zero while preserving
                # signed sensor noise and actual dynamic acceleration.
                dynamic = raw - raw.mean(axis=0)
                update['waveform'] = [
                    {'timestamp_ms': sample.timestamp_ms,
                     'x_g': round(float(dynamic[index, 0]), 5),
                     'y_g': round(float(dynamic[index, 1]), 5),
                     'z_g': round(float(dynamic[index, 2]), 5)}
                    for index, sample in enumerate(bridge.latest_samples)
                    if latest_ms - sample.timestamp_ms <= 5000
                ]
            seizure_trigger = False
            seizure_detected = False
            motion_status = None
            fall_trigger = False
            if seizure_detector:
                seizure_result = seizure_detector.push_window(window, end_ms)
                motion_status = getattr(seizure_detector, 'status', None)
                if motion_status:
                    if motion_status.get('motion') == 'rhythmic':
                        if seizure_rhythm_candidate_start_ms is None:
                            seizure_rhythm_candidate_start_ms = (
                                motion_status.get('rhythmic_start_ms') or
                                end_ms - int(motion_status.get('rhythmic_seconds', 0) * 1000))
                        seizure_rhythm_lost_ms = None
                    elif seizure_rhythm_candidate_start_ms is not None:
                        if seizure_rhythm_lost_ms is None:
                            seizure_rhythm_lost_ms = end_ms
                        elif (end_ms - seizure_rhythm_lost_ms) % (2 ** 32) >= args.episode_recovery_seconds * 1000:
                            seizure_rhythm_candidate_start_ms = None
                            seizure_rhythm_lost_ms = None
                    probability = motion_status['probability']
                    score_text = 'waiting' if probability is None else f'{probability:.3f}'
                    print(f"Possible seizure-like movement check: motion={motion_status['motion']}; "
                          f"rhythmic={motion_status['rhythmic_seconds']:.2f}s; "
                          f"ML probability={score_text}; reason={motion_status['reason']}",
                          flush=True)
                    update['seizure_motion'] = motion_status.copy()
                if seizure_result is not None:
                    seizure_label, seizure_probability = seizure_result
                    seizure_detected = seizure_label == 'possible_seizure_like_movement'
                    print(f'Possible seizure-like movement: {seizure_label}; probability: '
                          f'{seizure_probability:.2f}', flush=True)
                    if seizure_detected and not seizure_was_detected:
                        seizure_trigger = True
                        if dashboard and incident.state != 'MONITORING':
                            dashboard.event('Possible seizure-like movement detected (experimental)')
                    seizure_was_detected = seizure_detected
                    update['seizure'] = {
                        'label': seizure_label,
                        'probability': seizure_probability,
                    }
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
                    update['tremor_spectrum'] = tremor_detector.spectrum()
                    update['tremor_spectrum'] = tremor_detector.spectrum()
            if episode_tracker and sensor_samples:
                rhythmic = bool(motion_status and motion_status.get('motion') == 'rhythmic')
                frequency_valid = bool(rhythmic and
                                       motion_status.get('motion_rms_g', 0) >= args.seizure_motion_rms_g and
                                       motion_status.get('peak_power_share', 0) >= args.seizure_peak_power_share and
                                       motion_status.get('band_power_share', 0) >= args.seizure_band_power_share)
                spectrum = tremor_detector.spectrum() if tremor_detector else []
                episode_tracker.observe(
                    sensor_samples, detected=seizure_detected, rhythmic=rhythmic,
                    rhythmic_seconds=(motion_status or {}).get('rhythmic_seconds', 0),
                    frequency_hz=(motion_status or {}).get('peak_hz'),
                    frequency_valid=frequency_valid,
                    motion_rms_g=(motion_status or {}).get('motion_rms_g'),
                    window_rms_g=window_motion_rms, spectrum=spectrum,
                    source_type='simulated' if args.simulate else 'arduino',
                    onset_timestamp_ms=seizure_rhythm_candidate_start_ms)
                update['episode_report'] = episode_tracker.summary()
                update['episode_history'] = episode_tracker.history_summaries()
            if detector:
                fall_label, fall_probability = detector.predict(window)
                fall_detected = fall_label == 'fall'
                fall_trigger = (fall_detected and not fall_was_detected and
                                detector.should_trigger())
                fall_was_detected = fall_detected
                update['fall'] = {'label': fall_label, 'probability': fall_probability}
                print(f'Status: {incident.state}', flush=True)
            candidate_kind = ('fall_and_seizure_like_motion' if fall_trigger and seizure_trigger
                              else 'fall' if fall_trigger else
                              'seizure_like_motion' if seizure_trigger else None)
            if candidate_kind and incident.state == 'MONITORING':
                voice_active = bool(voice and voice.owner_voice is not None and not enrolling)
                incident.duration = voice_duration if voice_active else args.confirmation_seconds
                if incident.candidate(defer_timer=True, kind=candidate_kind):
                    # Finish the spoken question before opening the microphone or
                    # starting the manual confirmation countdown.
                    speak(CONFIRMATION_PROMPT)
                    if voice_active:
                        if dashboard:
                            dashboard.publish(voice_status='Starting voice confirmation',
                                              voice_can_enroll=False)
                        voice_recording = True
                        threading.Thread(target=voice_confirmation,
                                         args=(voice, voice_results, incident, dashboard), daemon=True).start()
                    else:
                        incident.start_confirmation_window()
            if dashboard:
                dashboard.publish(incident=incident.state, incident_type=incident.kind, **update)
    except KeyboardInterrupt:
        print('Stopping.')
    if collection_errors:
        raise RuntimeError('Sensor collection failed') from collection_errors[0]


if __name__ == '__main__':
    main()
