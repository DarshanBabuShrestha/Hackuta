"""Seizure-like motion uses the same confirmation state machine as falls."""
import numpy as np

from backend import main as backend_main
from backend.caregiver_alerts import CaregiverAlerts
from backend.incident_manager import IncidentManager
from backend.seizure_detector import SeizureMotionDetector


def run_windows(detector, samples):
    results = []
    for index, start in enumerate(range(0, len(samples) - 31, 16)):
        result = detector.push_window(samples[start:start + 32], 620 + index * 320)
        if result is not None:
            results.append(result)
    return results


def test_stationary_board_suppresses_positive_model_prediction():
    detector = SeizureMotionDetector()

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, vector):
            return np.array([[0.1, 0.9]])

        def predict(self, vector):
            return np.array([1])

    detector.model = Model()
    samples = np.tile([0, 0, 1, 0, 0, 0], (720, 1)).astype(float)
    results = run_windows(detector, samples)
    assert results and all(result == ('normal_activity', 0.9) for result in results)
    assert detector.status['motion'] == 'stationary'
    assert detector.status['reason'] == 'stationary'


def test_motion_still_reaches_classifier():
    detector = SeizureMotionDetector()

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, vector):
            return np.array([[0.1, 0.9]])

        def predict(self, vector):
            return np.array([1])

    detector.model = Model()
    t = np.arange(720) / 50
    samples = np.column_stack((0.8 * np.sin(2 * np.pi * 3 * t),
                               np.zeros(len(t)), np.ones(len(t)),
                               np.zeros((len(t), 3))))
    results = run_windows(detector, samples)
    assert results[0] == ('possible_seizure_like_movement', 0.9)
    assert detector.status['rhythmic_seconds'] >= 5


def test_ordinary_movement_does_not_pass_rhythm_gate():
    detector = SeizureMotionDetector()
    t = np.arange(720) / 50
    samples = np.column_stack((0.8 * np.sin(2 * np.pi * 1 * t),
                               np.zeros(len(t)), np.ones(len(t)),
                               np.zeros((len(t), 3))))
    results = run_windows(detector, samples)
    assert results and all(result[0] == 'normal_activity' for result in results)
    assert detector.status['reason'] == 'non-rhythmic movement'


def test_sustained_shaking_reaches_saved_model_threshold():
    detector = SeizureMotionDetector()
    t = np.arange(720) / 50
    samples = np.column_stack((0.8 * np.sin(2 * np.pi * 3 * t),
                               np.zeros(len(t)), np.ones(len(t)),
                               np.zeros((len(t), 3))))
    results = run_windows(detector, samples)
    assert results[0][0] == 'possible_seizure_like_movement'
    assert results[0][1] >= detector.threshold


def test_brief_shaking_does_not_complete_five_seconds():
    detector = SeizureMotionDetector()
    t = np.arange(720) / 50
    shake = np.where((t >= 10) & (t < 12), 0.8 * np.sin(2 * np.pi * 3 * t), 0)
    samples = np.column_stack((shake, np.zeros(len(t)), np.ones(len(t)),
                               np.zeros((len(t), 3))))
    results = run_windows(detector, samples)
    assert results and all(result[0] == 'normal_activity' for result in results)
    assert detector.status['rhythmic_seconds'] == 0


def test_sustained_rhythm_requires_model_score():
    detector = SeizureMotionDetector(threshold=0.95)
    class Model:
        classes_ = np.array([0, 1])
        def predict_proba(self, vector):
            return np.array([[0.1, 0.9]])
    detector.model = Model()
    t = np.arange(720) / 50
    samples = np.column_stack((0.8 * np.sin(2 * np.pi * 3 * t),
                               np.zeros(len(t)), np.ones(len(t)),
                               np.zeros((len(t), 3))))
    results = run_windows(detector, samples)
    assert results and all(result == ('normal_activity', 0.9) for result in results)
    assert detector.status['reason'] == 'ML probability below threshold'


def test_seizure_incident_can_be_cancelled():
    messages = []
    incident = IncidentManager(emit=messages.append)
    assert incident.candidate(kind='seizure_like_motion')
    assert incident.kind == 'seizure_like_motion'
    assert incident.state == 'CONFIRMING'
    assert messages[0] == 'POSSIBLE SEIZURE-LIKE MOVEMENT DETECTED'
    assert incident.cancel()
    assert incident.state == 'MONITORING'
    assert incident.kind is None


def test_seizure_detector_starts_confirmation(monkeypatch, capsys):
    spoken = []
    def capture(message):
        spoken.append(message)
        return True
    monkeypatch.setattr(backend_main, 'speak', capture)
    class Detector:
        def __init__(self, **kwargs):
            pass

        def push_window(self, window, end_ms):
            return 'possible_seizure_like_movement', 1.0

    monkeypatch.setattr(backend_main, 'SeizureMotionDetector', Detector)
    monkeypatch.setattr(backend_main, 'keyboard_reader', lambda commands: None)
    monkeypatch.setattr(
        backend_main, 'replay',
        lambda path, bridge, speed: iter([[[0, 0, 1, 0, 0, 0]] * 32]),
    )
    backend_main.main([
        '--simulate', 'unused.csv', '--mode', 'seizure', '--no-web',
        '--confirmation-seconds', '0.01',
    ])
    output = capsys.readouterr().out
    assert 'POSSIBLE SEIZURE-LIKE MOVEMENT DETECTED' in output
    assert output.count('Are you okay? Please confirm') == 1
    assert spoken[0] == 'Are you okay? Please confirm'
    assert backend_main.re.sub(r'\s+', ' ', ' '.join(spoken[1:])) == backend_main.re.sub(
        r'\s+', ' ', backend_main.BYSTANDER_ASSISTANCE_MESSAGE)
    assert 'CAREGIVER ALERT REQUIRED' in output


def test_simultaneous_model_predictions_report_both(monkeypatch, capsys):
    monkeypatch.setattr(backend_main, 'speak', lambda message: None)

    class FallModel:
        def __init__(self, **kwargs):
            pass

        def predict(self, window):
            return 'fall', 0.9

        def should_trigger(self):
            return True

    class SeizureModel:
        def __init__(self, **kwargs):
            pass

        def push_window(self, window, end_ms):
            return 'possible_seizure_like_movement', 0.8

    monkeypatch.setattr(backend_main, 'FallDetector', FallModel)
    monkeypatch.setattr(backend_main, 'SeizureMotionDetector', SeizureModel)
    monkeypatch.setattr(backend_main, 'keyboard_reader', lambda commands: None)
    monkeypatch.setattr(
        backend_main, 'replay',
        lambda path, bridge, speed: iter([[[0, 0, 1, 0, 0, 0]] * 32]),
    )
    monkeypatch.setattr(
        backend_main, 'CaregiverAlerts',
        lambda emit: CaregiverAlerts(emit, call=lambda kind: 'CA789', timeout=.01),
    )
    backend_main.main([
        '--simulate', 'unused.csv', '--mode', 'all', '--no-web',
        '--confirmation-seconds', '0.01',
    ])
    output = capsys.readouterr().out
    assert 'POSSIBLE FALL AND SEIZURE-LIKE MOVEMENT DETECTED' in output
