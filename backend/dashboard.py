"""Small local dashboard and thread-safe WebSocket snapshot publisher."""
import asyncio
import hmac
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse


class Dashboard:
    def __init__(self, commands, source, enabled=None, voice_enabled=False,
                 caregiver_key=None, simulation_waiting=False):
        self.commands = commands
        self.caregiver_key = caregiver_key
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
            'fall': None,
            'tremor': None,
            'seizure': None,
            'seizure_motion': None,
            'windows': 0,
            'updated_at': None,
            'events': [],
            'caregiver_alert': None,
            'caregiver_connected': False,
            'simulation_waiting': simulation_waiting,
        }
        self.app = FastAPI()

        @self.app.get('/')
        async def index(request: Request):
            if request.client and request.client.host not in ('127.0.0.1', '::1'):
                return PlainTextResponse('Wearer dashboard is local only', status_code=403)
            return HTMLResponse(Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8'))

        @self.app.get('/caregiver')
        async def caregiver_page():
            return HTMLResponse(Path(__file__).with_name('caregiver.html').read_text(encoding='utf-8'))

        @self.app.websocket('/ws')
        async def stream(socket: WebSocket):
            if getattr(socket, 'client', None) and socket.client.host not in ('127.0.0.1', '::1'):
                await socket.close(code=1008)
                return
            await self._stream(socket, caregiver=False)

        @self.app.websocket('/caregiver/ws')
        async def caregiver_stream(socket: WebSocket):
            key = socket.query_params.get('key', '')
            if not self.caregiver_key or not hmac.compare_digest(key, self.caregiver_key):
                await socket.close(code=1008)
                return
            await self._stream(socket, caregiver=True)

    async def _stream(self, socket, caregiver):
        await socket.accept()
        loop = asyncio.get_running_loop()
        updates = asyncio.Queue(maxsize=1)
        with self.lock:
            self.clients.add((loop, updates, caregiver))
            initial = self.snapshot(caregiver)
        if caregiver:
            self.publish(caregiver_connected=True)
        await socket.send_json(initial)

        async def receive_commands():
            while True:
                message = await socket.receive_json()
                if not isinstance(message, dict):
                    continue
                command = message.get('command')
                if caregiver:
                    if command == 'acknowledge' and isinstance(message.get('alert_id'), str):
                        self.commands.put({'command': command, 'alert_id': message['alert_id']})
                elif command in ('cancel', 'help', 'enroll_voice', 'start_simulation'):
                    self.commands.put(command)

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
                self.clients.discard((loop, updates, caregiver))
                connected = any(client[2] for client in self.clients)
            if caregiver:
                self.publish(caregiver_connected=connected)

    def snapshot(self, caregiver=False):
        if caregiver:
            return {key: self.state[key] for key in ('caregiver_alert', 'caregiver_connected', 'updated_at')}
        return self.state.copy()

    def publish(self, **changes):
        with self.lock:
            self.state.update(changes)
            self.state['updated_at'] = datetime.now(timezone.utc).isoformat()
            clients = tuple(self.clients)
            owner_snapshot = self.snapshot()
            caregiver_snapshot = self.snapshot(caregiver=True)
        for loop, updates, caregiver in clients:
            snapshot = caregiver_snapshot if caregiver else owner_snapshot
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


def serve(dashboard, port, host='127.0.0.1'):
    import uvicorn
    uvicorn.run(dashboard.app, host=host, port=port, log_level='warning')
