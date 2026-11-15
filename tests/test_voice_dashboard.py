"""Voice enrollment is started from the dashboard without terminal input."""
import asyncio
import queue

from fastapi import WebSocketDisconnect

from backend.dashboard import Dashboard
from backend.main import enroll_voice


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
        dashboard = Dashboard(commands, 'COM9', voice_enabled=True)

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
        await socket.received.put({'command': 'enroll_voice'})
        await asyncio.sleep(.05)
        assert commands.get_nowait() == 'enroll_voice'
        await socket.received.put(None)
        await asyncio.wait_for(task, 2)

    asyncio.run(check())
