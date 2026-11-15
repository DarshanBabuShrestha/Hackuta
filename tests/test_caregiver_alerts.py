import asyncio
import io
import json
import queue
import time
import urllib.error
import urllib.parse
from pathlib import Path

from fastapi import WebSocketDisconnect

from backend.caregiver_alerts import CaregiverAlerts, call_caregiver
from backend.dashboard import Dashboard
from backend import main as backend_main


def test_caregiver_alert_plays_tone_and_voice_once_per_incident():
    html = (Path(__file__).parents[1] / 'backend' / 'caregiver.html').read_text(encoding='utf-8')
    assert "CAREGIVER_VOICE_ALERT = 'Assistance is required. Please check on Odysseus'" in html
    assert 'activeVoiceAlert = new SpeechSynthesisUtterance(CAREGIVER_VOICE_ALERT)' in html
    assert 'const voice = cinematicVoice();' in html
    assert 'activeVoiceAlert.rate = 0.82;' in html
    assert 'activeVoiceAlert.pitch = 0.7;' in html
    assert 'unlockCaregiverVoice();' in html
    new_alert_block = html.split("if (alert.status === 'WAITING' && notifiedId !== alert.id)", 1)[1]
    assert new_alert_block.index('tone();') < new_alert_block.index('setTimeout(voiceAlert, 400);')


def test_acknowledgment_before_deadline_prevents_call():
    now = [0]
    called = []
    alerts = CaregiverAlerts(lambda message: None, call=called.append, clock=lambda: now[0])
    assert alerts.start('fall')
    alert_id = alerts.alert['id']
    now[0] = 9.9
    assert not alerts.acknowledge('wrong-id')
    assert alerts.acknowledge(alert_id)
    now[0] = 20
    alerts.tick()
    assert alerts.alert['status'] == 'ACKNOWLEDGED'
    assert not called
    assert alerts.start('fall')
    assert alerts.alert['id'] != alert_id
    assert not alerts.acknowledge(alert_id)


def test_timeout_requests_one_call_and_rejects_late_acknowledgment():
    now = [0]
    called = []
    alerts = CaregiverAlerts(lambda message: None,
                             call=lambda kind: called.append(kind) or 'CA123',
                             clock=lambda: now[0])
    alerts.start('seizure_like_motion')
    alert_id = alerts.alert['id']
    now[0] = 10
    assert not alerts.acknowledge(alert_id)
    for _ in range(100):
        alerts.tick()
        if alerts.alert['status'] == 'CALL_QUEUED':
            break
        time.sleep(.001)
    assert called == ['seizure_like_motion']
    assert alerts.alert['status'] == 'CALL_QUEUED'
    assert alerts.alert['detail'] == 'Call initiated'
    assert alerts.alert['call_sid'] == 'CA123'
    assert not alerts.acknowledge(alert_id)


def test_call_uses_configured_number_and_trial_compatible_twiml_url(monkeypatch):
    for name, value in {
        'TWILIO_ACCOUNT_SID': 'AC123',
        'TWILIO_AUTH_TOKEN': 'secret',
        'TWILIO_FROM_NUMBER': '+15550001111',
        'ODYSSEY_CAREGIVER_PHONE': '+15550002222',
    }.items():
        monkeypatch.setenv(name, value)
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            return json.dumps({'sid': 'CA456'}).encode()

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return Response()

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    assert call_caregiver('fall') == 'CA456'
    request, timeout = captured[0]
    body = urllib.parse.parse_qs(request.data.decode())
    assert request.full_url.endswith('/Accounts/AC123/Calls.json')
    assert body['To'] == ['+15550002222']
    assert body['From'] == ['+15550001111']
    assert body['Url'] == ['https://demo.twilio.com/docs/voice.xml']
    assert 'Twiml' not in body
    assert timeout == 8


def test_call_uses_custom_twiml_url(monkeypatch):
    for name, value in {
        'TWILIO_ACCOUNT_SID': 'AC123',
        'TWILIO_AUTH_TOKEN': 'secret',
        'TWILIO_FROM_NUMBER': '+15550001111',
        'ODYSSEY_CAREGIVER_PHONE': '+15550002222',
        'TWILIO_TWIML_URL': 'https://handler.twilio.com/twiml/example',
    }.items():
        monkeypatch.setenv(name, value)
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            return json.dumps({'sid': 'CA456'}).encode()

    def fake_urlopen(request, timeout):
        captured.append(request)
        return Response()

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    assert call_caregiver('fall') == 'CA456'
    body = urllib.parse.parse_qs(captured[0].data.decode())
    assert body['Url'] == ['https://handler.twilio.com/twiml/example']
    assert 'Twiml' not in body


def test_call_surfaces_twilio_trial_account_error(monkeypatch):
    for name, value in {
        'TWILIO_ACCOUNT_SID': 'AC123',
        'TWILIO_AUTH_TOKEN': 'secret',
        'TWILIO_FROM_NUMBER': '+15550001111',
        'ODYSSEY_CAREGIVER_PHONE': '+15550002222',
    }.items():
        monkeypatch.setenv(name, value)

    def reject_call(request, timeout):
        payload = json.dumps({
            'code': 21219,
            'message': 'The number is unverified. Trial accounts may only make calls to verified numbers.',
        }).encode()
        raise urllib.error.HTTPError(
            request.full_url, 400, 'Bad Request', {}, io.BytesIO(payload)
        )

    monkeypatch.setattr('urllib.request.urlopen', reject_call)
    try:
        call_caregiver('fall')
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError('Expected Twilio rejection')
    assert 'HTTP 400, Twilio code 21219' in message
    assert 'Trial accounts may only make calls to verified numbers' in message


def test_caregiver_websocket_requires_key_and_only_accepts_acknowledgment():
    async def check():
        commands = queue.Queue()
        dashboard = Dashboard(commands, 'COM9', caregiver_key='test-key')
        route = next(route for route in dashboard.app.routes
                     if getattr(route, 'path', None) == '/caregiver/ws')

        class Socket:
            def __init__(self, key):
                self.query_params = {'key': key}
                self.sent = asyncio.Queue()
                self.received = asyncio.Queue()
                self.closed = None

            async def close(self, code):
                self.closed = code

            async def accept(self):
                pass

            async def send_json(self, data):
                await self.sent.put(data)

            async def receive_json(self):
                message = await self.received.get()
                if message is None:
                    raise WebSocketDisconnect()
                return message

        bad = Socket('wrong')
        await route.endpoint(bad)
        assert bad.closed == 1008
        good = Socket('test-key')
        task = asyncio.create_task(route.endpoint(good))
        initial = await asyncio.wait_for(good.sent.get(), 1)
        assert set(initial) == {'caregiver_alert', 'caregiver_connected', 'location',
                                'location_status', 'updated_at'}
        await good.received.put({'command': 'cancel'})
        await good.received.put({'command': 'acknowledge', 'alert_id': 'incident-1'})
        await asyncio.sleep(.05)
        assert commands.get_nowait() == {'command': 'acknowledge', 'alert_id': 'incident-1'}
        assert commands.empty()
        await good.received.put(None)
        await asyncio.wait_for(task, 1)

    asyncio.run(check())


def test_backend_detects_second_fall_after_first_alert_finishes(monkeypatch, capsys):
    monkeypatch.setattr(backend_main, 'speak', lambda message: None)

    class Detector:
        def __init__(self, **kwargs):
            pass

        def predict(self, window):
            return ('fall', 1.0) if window[0][0] else ('adl', 0.0)

        def should_trigger(self):
            return True

    def replay(_path, _bridge, _speed):
        yield [[1, 0, 1, 0, 0, 0]] * 32
        yield [[0, 0, 1, 0, 0, 0]] * 32
        time.sleep(.5)
        yield [[1, 0, 1, 0, 0, 0]] * 32

    monkeypatch.setattr(backend_main, 'FallDetector', Detector)
    monkeypatch.setattr(backend_main, 'replay', replay)
    monkeypatch.setattr(backend_main, 'keyboard_reader', lambda commands: None)
    monkeypatch.setattr(
        backend_main, 'CaregiverAlerts',
        lambda emit: CaregiverAlerts(emit, call=lambda kind: 'CA789', timeout=.02),
    )
    backend_main.main(['--simulate', 'unused.csv', '--mode', 'fall', '--no-web',
                       '--cooldown', '0',
                       '--confirmation-seconds', '0.01'])
    output = capsys.readouterr().out
    assert output.count('FALL DETECTED') == 2
    assert output.count('CAREGIVER ALERT REQUIRED') == 2
    assert output.count('Caregiver alert cycle finished; monitoring resumed') == 2
