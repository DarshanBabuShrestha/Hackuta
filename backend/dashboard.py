"""Small local dashboard and thread-safe WebSocket snapshot publisher."""
import asyncio
import csv
import hmac
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response


class Dashboard:
    def __init__(self, commands, source, enabled=None, voice_enabled=False,
                 caregiver_key=None, simulation_waiting=False, episode_tracker=None):
        self.commands = commands
        self.caregiver_key = caregiver_key
        self.episode_tracker = episode_tracker
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
            'tremor_spectrum': [],
            'waveform': [],
            'seizure': None,
            'seizure_motion': None,
            'windows': 0,
            'updated_at': None,
            'events': [],
            'caregiver_alert': None,
            'caregiver_connected': False,
            'broadcast_active': False,
            'broadcast_paused': False,
            'location': None,
            'location_status': 'Location unavailable',
            'simulation_waiting': simulation_waiting,
            'episode_report': episode_tracker.summary() if episode_tracker else None,
            'episode_history': episode_tracker.history_summaries() if episode_tracker else [],
        }
        self.app = FastAPI()

        @self.app.get('/')
        async def index(request: Request):
            if request.client and request.client.host not in ('127.0.0.1', '::1'):
                return PlainTextResponse('Wearer dashboard is local only', status_code=403)
            return HTMLResponse(Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8'))

        @self.app.get('/assets/odyssey-header.webp')
        async def odyssey_header_art():
            return FileResponse(Path(__file__).with_name('assets') / 'odyssey-header.webp',
                                media_type='image/webp')

        def owner_local(request):
            return bool(request.client and request.client.host in ('127.0.0.1', '::1', 'localhost'))

        @self.app.get('/api/episodes')
        async def episode_history(request: Request):
            if not owner_local(request):
                return PlainTextResponse('Episode reports are available on the wearer computer only', status_code=403)
            if not self.episode_tracker:
                return {'active': None, 'reports': []}
            return {'active': self.episode_tracker.summary(),
                    'reports': self.episode_tracker.history_summaries()}

        @self.app.get('/api/episodes/{episode_id}')
        async def episode_report(episode_id: str, request: Request):
            if not owner_local(request):
                return PlainTextResponse('Episode reports are available on the wearer computer only', status_code=403)
            if len(episode_id) != 32 or any(character not in '0123456789abcdef' for character in episode_id):
                return PlainTextResponse('Episode report not found', status_code=404)
            report = self.episode_tracker.get_report(episode_id) if self.episode_tracker else None
            if report is None:
                return PlainTextResponse('Episode report not found', status_code=404)
            return JSONResponse(report)

        @self.app.get('/api/episodes/{episode_id}/export')
        async def export_episode(episode_id: str, request: Request, format: str = 'json'):
            if not owner_local(request):
                return PlainTextResponse('Episode reports are available on the wearer computer only', status_code=403)
            if len(episode_id) != 32 or any(character not in '0123456789abcdef' for character in episode_id):
                return PlainTextResponse('Episode report not found', status_code=404)
            report = self.episode_tracker.get_report(episode_id) if self.episode_tracker else None
            if report is None:
                return PlainTextResponse('Episode report not found', status_code=404)
            filename = f"odyssey-episode-{episode_id}"
            if format.lower() == 'json':
                return Response(json.dumps(report, indent=2), media_type='application/json',
                                headers={'Content-Disposition': f'attachment; filename="{filename}.json"'})
            if format.lower() != 'csv':
                return PlainTextResponse('format must be json or csv', status_code=422)
            output = io.StringIO(newline='')
            fields = ['record_type', 'episode_id', 'source_type', 'start_timestamp_ms',
                      'start_elapsed_ms', 'end_timestamp_ms', 'end_elapsed_ms',
                      'duration_seconds', 'average_frequency_hz',
                      'highest_dominant_frequency_hz', 'frequency_variability_hz',
                      'peak_movement_g_rms', 'average_movement_g_rms',
                      'above_average_bursts', 'time_above_average_seconds',
                      'peak_movement_timestamp_ms', 'sample_count', 'timestamp_ms', 'elapsed_ms',
                      'recording_start_timestamp_ms', 'recording_end_timestamp_ms',
                      'missing_data', 'data_quality_warnings', 'movement_intensity_method',
                      'frequency_method', 'peak_spectrum_timestamp_ms',
                      'phase', 'x_g', 'y_g', 'z_g', 'dynamic_x_g', 'dynamic_y_g',
                      'dynamic_z_g', 'movement_g', 'frequency_hz', 'frequency_valid',
                      'movement_rms_g', 'spectrum_json']
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            timeline = {item['elapsed_ms']: item for item in report.get('frequency_timeline', [])}
            summary = {key: report.get(key) for key in fields
                       if key in report and key not in ('record_type', 'timestamp_ms', 'elapsed_ms')}
            for sample in report.get('samples', []):
                frequency = timeline.get(sample.get('elapsed_ms'), {})
                row = dict(summary, record_type='sensor_sample', timestamp_ms=sample['timestamp_ms'],
                           elapsed_ms=sample.get('elapsed_ms'))
                row.update({key: sample.get(key) for key in
                            ('phase', 'x_g', 'y_g', 'z_g', 'dynamic_x_g', 'dynamic_y_g',
                             'dynamic_z_g', 'movement_g')})
                row.update({key: frequency.get(key) for key in
                            ('frequency_hz', 'frequency_valid', 'movement_rms_g')})
                row['spectrum_json'] = (json.dumps(report.get('peak_spectrum'))
                                        if report.get('peak_spectrum_timestamp_ms') == sample['timestamp_ms']
                                        else '')
                writer.writerow(row)
            return Response(output.getvalue(), media_type='text/csv',
                            headers={'Content-Disposition': f'attachment; filename="{filename}.csv"'})

        @self.app.get('/caregiver')
        async def caregiver_page():
            return HTMLResponse(Path(__file__).with_name('caregiver.html').read_text(encoding='utf-8'))

        @self.app.post('/api/location')
        async def update_location(request: Request):
            supplied_key = request.headers.get('x-caregiver-key', '')
            if not self.caregiver_key or not hmac.compare_digest(supplied_key, self.caregiver_key):
                return PlainTextResponse('Unauthorized', status_code=401)
            host = request.url.hostname
            local = host in ('127.0.0.1', 'localhost', '::1')
            forwarded_proto = request.headers.get('x-forwarded-proto', '').split(',')[0].strip()
            if not local and request.url.scheme != 'https' and forwarded_proto != 'https':
                return PlainTextResponse('HTTPS required', status_code=403)
            try:
                data = await request.json()
                latitude, longitude = float(data['latitude']), float(data['longitude'])
                accuracy, timestamp = float(data['accuracy']), str(data['timestamp'])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and
                        0 <= accuracy <= 100000 and len(timestamp) <= 64):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return PlainTextResponse('Invalid location payload', status_code=422)
            self.publish(location={'latitude': latitude, 'longitude': longitude,
                                   'accuracy': accuracy, 'timestamp': timestamp},
                         location_status='Available')
            return {'ok': True}

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
                elif command in ('cancel', 'help', 'enroll_voice', 'start_simulation',
                                 'pause_broadcast'):
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
            return {key: self.state[key] for key in ('caregiver_alert', 'caregiver_connected',
                    'location', 'location_status', 'updated_at')}
        return {key: value for key, value in self.state.items()
                if key not in ('location', 'location_status')}

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
