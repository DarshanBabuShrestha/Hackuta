"""Small local dashboard and thread-safe WebSocket snapshot publisher."""
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


class Dashboard:
    def __init__(self, commands, source, enabled=None, voice_enabled=False):
        self.commands = commands
        self.lock = threading.Lock()
        self.clients = set()
        self.state = {
            'source': source,
            'enabled': enabled or {'fall': True, 'tremor': True, 'seizure': True},
            'voice_enabled': voice_enabled,
            'voice_status': 'Loading voice model' if voice_enabled else 'Off',
            'voice_can_enroll': False,
            'voice_enrolled': False,
            'voice_device': None,
            'sensor_status': 'Waiting for connection',
            'incident': 'MONITORING',
            'incident_type': None,
            'motion': None,
            'tremor': None,
            'seizure': None,
            'windows': 0,
            'updated_at': None,
            'events': [],
        }
        self.app = FastAPI()

        @self.app.get('/')
        async def index():
            return HTMLResponse(Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8'))

        @self.app.websocket('/ws')
        async def stream(socket: WebSocket):
            await socket.accept()
            loop = asyncio.get_running_loop()
            updates = asyncio.Queue(maxsize=1)
            with self.lock:
                self.clients.add((loop, updates))
                initial = self.state.copy()
            await socket.send_json(initial)

            async def receive_commands():
                while True:
                    message = await socket.receive_json()
                    if isinstance(message, dict) and message.get('command') in ('cancel', 'help', 'enroll_voice'):
                        self.commands.put(message['command'])

            receiver = asyncio.create_task(receive_commands())
            try:
                while True:
                    update = asyncio.create_task(updates.get())
                    done, pending = await asyncio.wait(
                        (update, receiver), return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        if task is update:
                            task.cancel()
                    if receiver in done:
                        receiver.result()
                        break
                    await socket.send_json(update.result())
            except WebSocketDisconnect:
                pass
            finally:
                receiver.cancel()
                with self.lock:
                    self.clients.discard((loop, updates))

    def publish(self, **changes):
        with self.lock:
            self.state.update(changes)
            self.state['updated_at'] = datetime.now(timezone.utc).isoformat()
            snapshot = self.state.copy()
            clients = tuple(self.clients)
        for loop, updates in clients:
            def offer(queue=updates, value=snapshot):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(value)
            try:
                loop.call_soon_threadsafe(offer)
            except RuntimeError:
                # A client may close between taking the snapshot and posting it.
                pass

    def event(self, message):
        with self.lock:
            events = self.state['events']
            self.state['events'] = ([{
                'time': datetime.now().astimezone().strftime('%H:%M:%S'),
                'message': message,
            }] + events)[:8]
        self.publish()


def serve(dashboard, port):
    import uvicorn
    uvicorn.run(dashboard.app, host='127.0.0.1', port=port, log_level='warning')
