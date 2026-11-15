"""Voice enrollment is started from the dashboard without terminal input."""
import asyncio
import queue
import re

from fastapi import WebSocketDisconnect

from backend.dashboard import Dashboard
from backend.main import (BYSTANDER_ASSISTANCE_MESSAGE, VOICE_REJECTED_MESSAGE,
                          broadcast_first_aid_message, enroll_voice,
                          reject_voice_confirmation, toggle_first_aid_broadcast)


def test_bystander_first_aid_message_is_spoken(monkeypatch):
    spoken = []
    updates = []
    def capture(message):
        spoken.append(message)
        return True
    monkeypatch.setattr('backend.main.speak', capture)

    class View:
        def publish(self, **changes):
            updates.append(changes)

    broadcast_first_aid_message(View())

    assert re.sub(r'\s+', ' ', ' '.join(spoken)) == re.sub(r'\s+', ' ', BYSTANDER_ASSISTANCE_MESSAGE)
    assert updates == [
        {'broadcast_active': True, 'broadcast_paused': False},
        {'broadcast_active': False, 'broadcast_paused': False},
    ]
    assert BYSTANDER_ASSISTANCE_MESSAGE.startswith('Hello! I may be unconscious')
    assert 'Do not hold me down or put anything in my mouth.' in BYSTANDER_ASSISTANCE_MESSAGE
    assert 'Call 911 immediately' in BYSTANDER_ASSISTANCE_MESSAGE
    assert BYSTANDER_ASSISTANCE_MESSAGE.endswith('until I recover or help arrives.')


def test_broadcast_button_toggles_pause_and_resume(monkeypatch):
    updates, events, stopped = [], [], []

    class View:
        def publish(self, **changes):
            updates.append(changes)
        def event(self, message):
            events.append(message)

    monkeypatch.setattr('backend.main.stop_speaking', lambda: stopped.append(True))
    from backend import main as backend_main
    backend_main._broadcast_active.set()
    backend_main._broadcast_can_play.set()
    try:
        assert toggle_first_aid_broadcast(View())
        assert not backend_main._broadcast_can_play.is_set()
        assert toggle_first_aid_broadcast(View())
        assert backend_main._broadcast_can_play.is_set()
    finally:
        backend_main._broadcast_active.clear()
        backend_main._broadcast_can_play.set()

    assert stopped == [True]
    assert updates == [
        {'broadcast_active': True, 'broadcast_paused': True},
        {'broadcast_active': True, 'broadcast_paused': False},
    ]
    assert events == ['Emergency broadcast paused by wearer',
                      'Emergency broadcast resumed by wearer']


def test_unrecognized_voice_alerts_caregiver_once(capsys, monkeypatch):
    updates = []
    spoken = []
    monkeypatch.setattr('backend.main.speak', spoken.append)

    class Incident:
        def __init__(self):
            self.alerts = 0

        def caregiver_pending(self):
            self.alerts += 1

    class View:
        def publish(self, **changes):
            updates.append(changes)

    incident = Incident()
    reject_voice_confirmation(incident, View())

    assert capsys.readouterr().out.strip() == VOICE_REJECTED_MESSAGE
    assert spoken == [VOICE_REJECTED_MESSAGE]
    assert incident.alerts == 1
    assert updates == [{'voice_status': VOICE_REJECTED_MESSAGE}]


def test_enrollment_uses_dashboard_command_without_terminal_prompt():
    events = queue.Queue()
    updates = []

    class View:
        def publish(self, **changes):
            updates.append(changes)

    class Verifier:
        def enroll(self, wait_for_enter, on_started, on_recorded):
            assert wait_for_enter is False
            on_started()
            on_recorded()

    enroll_voice(Verifier(), events, View())
    assert events.get_nowait()[0] == 'enrolled'
    assert [item['voice_status'] for item in updates] == [
        'Recording enrollment for 4 seconds', 'Saving voice enrollment'
    ]


def test_websocket_accepts_enrollment_command():
    async def check():
        commands = queue.Queue()
        dashboard = Dashboard(commands, 'COM9', voice_enabled=True, simulation_waiting=True)

        class Socket:
            def __init__(self):
                self.sent = asyncio.Queue()
                self.received = asyncio.Queue()

            async def accept(self):
                pass

            async def send_json(self, data):
                await self.sent.put(data)

            async def receive_json(self):
                message = await self.received.get()
                if message is None:
                    raise WebSocketDisconnect()
                return message

        socket = Socket()
        route = next(route for route in dashboard.app.routes if getattr(route, 'path', None) == '/ws')
        task = asyncio.create_task(route.endpoint(socket))
        initial = await asyncio.wait_for(socket.sent.get(), 2)
        assert initial['voice_enabled'] and not initial['voice_can_enroll']
        assert initial['simulation_waiting']
        await socket.received.put({'command': 'enroll_voice'})
        await asyncio.sleep(.05)
        assert commands.get_nowait() == 'enroll_voice'
        await socket.received.put({'command': 'start_simulation'})
        await asyncio.sleep(.05)
        assert commands.get_nowait() == 'start_simulation'
        await socket.received.put({'command': 'pause_broadcast'})
        await asyncio.sleep(.05)
        assert commands.get_nowait() == 'pause_broadcast'
        await socket.received.put(None)
        await asyncio.wait_for(task, 2)

    asyncio.run(check())
