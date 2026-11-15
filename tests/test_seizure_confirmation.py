"""Seizure-like motion uses the same confirmation state machine as falls."""
from backend import main as backend_main
from backend.incident_manager import IncidentManager


def test_seizure_incident_can_be_cancelled():
    messages = []
    incident = IncidentManager(emit=messages.append)
    assert incident.candidate(kind='seizure_like_motion')
    assert incident.kind == 'seizure_like_motion'
    assert incident.state == 'CONFIRMING'
    assert messages[0] == 'SEIZURE-LIKE MOTION DETECTED'
    assert incident.cancel()
    assert incident.state == 'MONITORING'
    assert incident.kind is None


def test_seizure_detector_starts_confirmation(monkeypatch, capsys):
    class Detector:
        def __init__(self, threshold):
            pass

        def push_window(self, window, end_ms):
            return 'simulated_seizure_like_motion', 1.0

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
    assert 'SEIZURE-LIKE MOTION DETECTED' in output
    assert 'Are you okay?' in output
    assert 'CAREGIVER ALERT REQUIRED' in output
