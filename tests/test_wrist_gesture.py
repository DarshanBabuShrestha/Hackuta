from backend.incident_manager import IncidentManager
from backend import main as backend_main
from backend.caregiver_alerts import CaregiverAlerts
from backend.dashboard import Dashboard
from backend.serial_bridge import Sample
from backend.wrist_gesture import WristGestureConfig, WristGestureRecognizer


def sample(timestamp_ms, gx=0, gy=0, gz=0):
    return Sample(timestamp_ms, (0.0, 0.0, 1.0, gx, gy, gz))


def run_segments(segments, *, axis=0, confirming=True, seizure_active=False, config=None):
    gesture = WristGestureRecognizer(config or WristGestureConfig())
    axis_index = 3 + axis
    timestamp = 0
    stages = []
    fired_at = None
    for speed, duration_seconds in segments:
        samples = round(duration_seconds * 50)
        for _ in range(samples):
            timestamp += 20
            values = [0.0] * 6
            values[2] = 1.0
            values[axis_index] = speed
            fired = gesture.feed(Sample(timestamp, tuple(values)),
                                 confirming=confirming,
                                 seizure_active=seizure_active)
            if not stages or stages[-1] != gesture.status['stage']:
                stages.append(gesture.status['stage'])
            if fired and fired_at is None:
                fired_at = timestamp
    return gesture, stages, fired_at


def valid_pattern(direction=1, axis=0):
    return [(0, .1), (direction * 240, .4), (0, .1),
            (-direction * 240, .4), (0, 1.2)]


def test_signed_peak_sequence_transitions_and_confirms_after_continuous_stillness():
    gesture, stages, fired_at = run_segments(valid_pattern())
    assert stages == ['WAITING', 'FIRST_TWIST', 'OPPOSITE_TWIST', 'STILLNESS', 'CONFIRMED']
    assert fired_at is not None
    assert gesture.status['stage'] == 'CONFIRMED'
    assert gesture.status['stillness_seconds'] >= 1.0
    assert [marker['stage'] for marker in gesture.status['markers']][-4:] == [
        'FIRST_TWIST', 'OPPOSITE_TWIST', 'STILLNESS', 'CONFIRMED']


def test_reverse_signed_peak_order_and_y_axis_are_supported():
    gesture, stages, fired_at = run_segments(valid_pattern(direction=-1), axis=1)
    assert fired_at is not None
    assert stages[-1] == 'CONFIRMED'
    assert gesture.status['axis'] == 'y'


def test_insufficient_angle_is_reported_and_does_not_confirm():
    gesture, _, fired_at = run_segments([
        (0, .1), (240, .06), (0, .1), (-240, .4), (0, 1.2)])
    assert fired_at is None
    assert any('Insufficient angle' in marker.get('reason', '')
               for marker in gesture.status['markers'])


def test_same_direction_second_lobe_is_rejected():
    gesture, _, fired_at = run_segments([
        (0, .1), (240, .35), (0, .2), (240, .35), (0, 1.2)])
    assert fired_at is None
    assert any('Wrong direction' in marker.get('reason', '')
               for marker in gesture.status['markers'])


def test_opposite_twist_timeout_is_reported():
    gesture, _, fired_at = run_segments([
        (0, .1), (240, .25), (0, 1.8), (-240, .4), (0, 1.2)])
    assert fired_at is None
    assert 'Timeout' in gesture.status['rejection_reason']


def test_stillness_must_be_uninterrupted_for_one_second():
    gesture, _, fired_at = run_segments([
        (0, .1), (240, .4), (0, .1), (-240, .4),
        (0, .45), (180, .08), (0, .75)])
    assert fired_at is None
    assert gesture.status['stage'] == 'STILLNESS'
    assert ('Insufficient stillness' in gesture.status['reason'] or
            'Insufficient stillness' in gesture.status['rejection_reason'])


def test_single_sample_impact_does_not_create_two_rotation_peaks():
    gesture = WristGestureRecognizer()
    fired = False
    for timestamp, speed in enumerate([0, 0, 0, 500, 0, 0, 0] + [0] * 60, start=1):
        fired = gesture.feed(sample(timestamp * 20, gx=speed), confirming=True) or fired
    assert not fired
    assert gesture.status['stage'] != 'CONFIRMED'


def test_confirmation_state_and_seizure_episode_blocks_are_enforced():
    gesture, _, fired_at = run_segments(valid_pattern(), confirming=False)
    assert fired_at is None
    assert gesture.status['reason'] == 'Armed; gesture detection starts during CONFIRMING'
    gesture, _, fired_at = run_segments(valid_pattern(), seizure_active=True)
    assert fired_at is None
    assert gesture.status['reason'] == 'Blocked while seizure-like movement or an episode is active'


def test_replay_source_is_disabled_even_during_confirmation():
    gesture = WristGestureRecognizer()
    timestamp = 0
    for speed, duration in valid_pattern():
        for _ in range(round(duration * 50)):
            timestamp += 20
            assert not gesture.feed(sample(timestamp, gx=speed), confirming=True,
                                    live_sensor=False)
    assert gesture.status['reason'] == 'Replay data cannot cancel incidents'


def test_serial_gap_rejects_pending_pattern():
    gesture = WristGestureRecognizer()
    for timestamp in range(20, 260, 20):
        gesture.feed(sample(timestamp, gx=240), confirming=True)
    gesture.feed(sample(1000), confirming=True)
    assert gesture.status['stage'] == 'WAITING'
    assert 'Timeout: serial sample gap' in gesture.status['rejection_reason']


def test_gesture_cancellation_logs_the_requested_message():
    messages = []
    incident = IncidentManager(emit=messages.append)
    assert incident.candidate()
    assert incident.cancel(source='wrist gesture')
    assert messages[-1] == 'Cancelled by wrist gesture; status: MONITORING'


def test_dashboard_shows_armed_instead_of_waiting_when_idle():
    dashboard = Dashboard(commands=None, source='COM-test')
    assert dashboard.state['wrist_gesture']['stage'] == 'ARMED'
    assert 'CONFIRMING' in dashboard.state['wrist_gesture']['reason']


def test_replayed_sensor_samples_cannot_cancel_an_incident(monkeypatch, capsys):
    seen_feed_gates = []

    class SpyGesture:
        def __init__(self, config):
            self.detector = WristGestureRecognizer(config)
            self.status = self.detector.status

        def feed(self, sensor_sample, *, confirming, seizure_active=False, live_sensor=True):
            seen_feed_gates.append((confirming, live_sensor))
            fired = self.detector.feed(sensor_sample, confirming=confirming,
                                       seizure_active=seizure_active,
                                       live_sensor=live_sensor)
            self.status = self.detector.status
            return fired

    class SeizureDetector:
        def __init__(self, **kwargs):
            self.status = {'motion': 'stationary', 'probability': .99,
                           'rhythmic_seconds': 0.0, 'motion_rms_g': 0.0,
                           'peak_power_share': 0.0, 'band_power_share': 0.0,
                           'reason': 'test fixture'}
            self.windows = 0

        def push_window(self, window, end_ms):
            self.windows += 1
            return (('possible_seizure_like_movement', .99) if self.windows == 1
                    else ('normal_activity', .01))

    def replay_recording(path, bridge, speed):
        phases = [(0, .1), (240, .4), (0, .1), (-240, .4), (0, 1.4)]
        values = []
        for gyro, duration in phases:
            values.extend([gyro] * round(duration * 50))
        for index, gyro_x in enumerate(values):
            sensor_sample = Sample(index * 20, (0, 0, 1, gyro_x, 0, 0))
            window = bridge.push(sensor_sample)
            if window is not None:
                yield window

    monkeypatch.setattr(backend_main, 'WristGestureRecognizer', SpyGesture)
    monkeypatch.setattr(backend_main, 'SeizureMotionDetector', SeizureDetector)
    monkeypatch.setattr(backend_main, 'keyboard_reader', lambda commands: None)
    monkeypatch.setattr(backend_main, 'replay', replay_recording)
    monkeypatch.setattr(
        backend_main, 'CaregiverAlerts',
        lambda emit: CaregiverAlerts(emit, call=lambda kind: 'CA789', timeout=.01),
    )
    backend_main.main(['--simulate', 'unused.csv', '--mode', 'seizure', '--no-web',
                       '--confirmation-seconds', '0.01'])
    output = capsys.readouterr().out
    assert seen_feed_gates
    assert all(not live_sensor for _, live_sensor in seen_feed_gates)
    assert 'Cancelled by wrist gesture' not in output
