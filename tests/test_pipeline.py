import numpy as np
import pytest

from backend.features import FEATURE_NAMES, G, WINDOW_SIZE, extract_features, extract_wire_features, wire_to_si
from backend.fall_detector import FallDetector
from backend.incident_manager import IncidentManager
from backend.serial_bridge import SerialBridge, parse_message


def test_parse_and_reject_bad_messages():
    sample = parse_message(b'IMU,20,0,0,1,0,90,0\r\n')
    assert sample.timestamp_ms == 20
    assert sample.values == (0, 0, 1, 0, 90, 0)
    for line in ('HEADER,t_ms,ax,ay,az,gx,gy,gz', 'IMU,2,1,2',
                 'IMU,2,nan,0,0,0,0,0', 'IMU,-1,0,0,0,0,0,0',
                 'IMU,2,33,0,0,0,0,0'):
        with pytest.raises(ValueError):
            parse_message(line)


def test_units_features_and_order():
    wire = np.tile([0, 0, 1, 0, 180, 0], (WINDOW_SIZE, 1)).astype(float)
    si = wire_to_si(wire)
    assert si[0, 2] == pytest.approx(G)
    assert si[0, 4] == pytest.approx(np.pi)
    assert len(FEATURE_NAMES) == 56
    assert FEATURE_NAMES[0] == 'acg_x_mean'
    assert FEATURE_NAMES[-1] == 'gyro_mag_mean_abs_step'
    features = extract_wire_features(wire)
    assert features[FEATURE_NAMES.index('acg_z_mean')] == pytest.approx(G)
    assert features[FEATURE_NAMES.index('gyro_y_mean')] == pytest.approx(np.pi)
    assert features[FEATURE_NAMES.index('acg_mag_median')] == pytest.approx(G)
    with pytest.raises(ValueError):
        extract_features(si[:-1])
    wire[0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_wire_features(wire)


def test_window_gap_and_hop():
    bridge = SerialBridge('test', hop=16)
    emitted = []
    for i in range(48):
        emitted.append(bridge.push(parse_message(f'IMU,{i*20},0,0,1,0,0,0')))
    assert sum(x is not None for x in emitted) == 2
    assert bridge.push(parse_message('IMU,5000,0,0,1,0,0,0')) is None
    assert len(bridge.buffer) == 1


def test_model_load_predict():
    detector = FallDetector()
    window = np.tile([0, 0, 1, 0, 0, 0], (WINDOW_SIZE, 1))
    label, probability = detector.predict(window)
    features = extract_wire_features(window).reshape(1, -1)
    assert label == ('fall' if probability >= detector.threshold else 'adl')
    assert probability == pytest.approx(detector.model.predict_proba(features)[0][list(detector.model.classes_).index('fall')])


def test_fall_threshold_and_cooldown():
    now = [0]
    detector = FallDetector(threshold=.7, cooldown=20, clock=lambda: now[0])

    class Model:
        classes_ = np.array(['adl', 'fall'])

        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, features):
            return np.array([[1 - self.probability, self.probability]])

    window = np.tile([0, 0, 1, 0, 0, 0], (WINDOW_SIZE, 1))
    detector.model = Model(.6)
    assert detector.predict(window)[0] == 'adl'
    detector.model = Model(.8)
    assert detector.predict(window)[0] == 'fall'
    assert detector.should_trigger()
    now[0] = 10
    assert not detector.should_trigger()
    now[0] = 20
    assert detector.should_trigger()


def test_incident_cancel_expire_and_duplicate():
    now = [0]
    output = []
    incident = IncidentManager(clock=lambda: now[0], emit=output.append)
    assert incident.candidate()
    assert not incident.candidate()
    assert incident.state == 'CONFIRMING'
    assert incident.cancel()
    assert incident.state == 'MONITORING'
    assert not incident.cancel()
    assert incident.candidate()
    now[0] = 16
    incident.tick()
    assert incident.state == 'CAREGIVER_PENDING'
    assert 'CAREGIVER ALERT REQUIRED' in output
    assert not incident.candidate()
    assert incident.close_caregiver()
    assert incident.state == 'MONITORING'
    assert incident.candidate(kind='fall')
    assert output.count('FALL DETECTED') == 3
