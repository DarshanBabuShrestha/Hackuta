"""Caregiver acknowledgment window and optional Twilio phone escalation."""
import base64
import json
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape


def call_caregiver(kind):
    """Queue a voice call with Twilio; return its call SID after HTTP acceptance."""
    names = ('TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_NUMBER',
             'ODYSSEY_CAREGIVER_PHONE')
    values = [os.environ.get(name, '').strip() for name in names]
    if not all(values):
        raise RuntimeError('Phone call is not configured; set ' +
                           ', '.join(name for name, value in zip(names, values) if not value))
    sid, token, from_number, to_number = values
    incident = ('a possible fall and seizure-like movement' if kind == 'fall_and_seizure_like_motion'
                else 'possible seizure-like movement' if kind == 'seizure_like_motion'
                else 'a possible fall')
    speech = f'Odyssey Guardian alert. The wearer may need immediate assistance after {incident}. Please check on them now.'
    body = urllib.parse.urlencode({
        'To': to_number,
        'From': from_number,
        'Twiml': f'<Response><Say>{escape(speech)}</Say></Response>',
    }).encode()
    request = urllib.request.Request(
        f'https://api.twilio.com/2010-04-01/Accounts/{urllib.parse.quote(sid)}/Calls.json',
        data=body,
        headers={
            'Authorization': 'Basic ' + base64.b64encode(f'{sid}:{token}'.encode()).decode(),
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.load(response)['sid']


class CaregiverAlerts:
    def __init__(self, emit, call=call_caregiver, clock=time.monotonic, timeout=10):
        self.emit, self.call, self.clock, self.timeout = emit, call, clock, timeout
        self.alert = None
        self.deadline = None
        self.results = queue.Queue()

    def start(self, kind):
        if self.alert and self.alert['status'] in ('WAITING', 'CALLING'):
            return False
        now = datetime.now(timezone.utc)
        self.alert = {
            'id': uuid.uuid4().hex,
            'kind': kind,
            'created_at': now.isoformat(),
            'response_deadline': (now + timedelta(seconds=self.timeout)).isoformat(),
            'status': 'WAITING',
            'detail': 'Waiting for caregiver acknowledgment',
        }
        self.deadline = self.clock() + self.timeout
        self.emit('Caregiver app alert available; awaiting acknowledgment')
        return True

    def acknowledge(self, alert_id):
        if not self.alert or self.alert['id'] != alert_id or self.alert['status'] != 'WAITING':
            return False
        # An acknowledgment at or after the deadline cannot suppress escalation.
        if self.clock() >= self.deadline:
            self.tick()
            return False
        self.alert['status'] = 'ACKNOWLEDGED'
        self.alert['detail'] = 'Caregiver acknowledged the alert'
        self.deadline = None
        self.emit('Caregiver acknowledged the alert')
        return True

    def tick(self):
        while not self.results.empty():
            alert_id, result, error = self.results.get_nowait()
            if self.alert and self.alert['id'] == alert_id and self.alert['status'] == 'CALLING':
                if error:
                    self.alert['status'] = 'CALL_FAILED'
                    self.alert['detail'] = f'Phone call failed: {error}'
                    self.emit(self.alert['detail'])
                else:
                    self.alert['status'] = 'CALL_QUEUED'
                    self.alert['detail'] = 'Phone provider accepted the call request'
                    self.alert['call_sid'] = result
                    self.emit('Phone provider accepted the caregiver call request')
        if self.alert and self.alert['status'] == 'WAITING' and self.clock() >= self.deadline:
            self.alert['status'] = 'CALLING'
            self.alert['detail'] = 'No caregiver response in 10 seconds; requesting phone call'
            self.deadline = None
            self.emit(self.alert['detail'])
            alert_id, kind = self.alert['id'], self.alert['kind']

            def make_call():
                try:
                    self.results.put((alert_id, self.call(kind), None))
                except Exception as exc:
                    self.results.put((alert_id, None, str(exc)))

            threading.Thread(target=make_call, daemon=True).start()

    @property
    def unfinished(self):
        return bool(self.alert and self.alert['status'] in ('WAITING', 'CALLING'))
