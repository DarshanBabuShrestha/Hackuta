"""Non-blocking incident confirmation state machine."""
import time

CAPTURE_COMPLETION_GRACE = 0.5


class IncidentManager:
    def __init__(self, duration=15, clock=time.monotonic, emit=print):
        self.duration, self.clock, self.emit = duration, clock, emit
        self.state = 'MONITORING'
        self.kind = None
        self.deadline = None

    def candidate(self, prompt=None, defer_timer=False, kind='fall'):
        if self.state != 'MONITORING':
            return False
        self.kind = kind
        self.state = 'CANDIDATE'
        label = ('POSSIBLE FALL AND SEIZURE-LIKE MOVEMENT' if kind == 'fall_and_seizure_like_motion'
                 else 'POSSIBLE SEIZURE-LIKE MOVEMENT' if kind == 'seizure_like_motion' else 'FALL')
        self.emit(f'{label} DETECTED')
        self.state = 'CONFIRMING'
        self.deadline = None if defer_timer else self.clock() + self.duration
        self.emit(prompt or 'Are you okay? Type c then Enter to cancel.')
        return True

    def start_response_window(self):
        if self.state != 'CONFIRMING':
            return False
        # Allow the audio callback to return control after its exact-length
        # capture before timing out the incident.
        self.deadline = self.clock() + self.duration + CAPTURE_COMPLETION_GRACE
        return True

    def response_captured(self):
        """Pause the response timeout while a captured answer is verified."""
        if self.state != 'CONFIRMING':
            return False
        self.state = 'VERIFYING'
        self.deadline = None
        self.emit('Voice captured; verifying response.')
        return True

    def cancel(self):
        if self.state not in ('CONFIRMING', 'VERIFYING'):
            return False
        self.state = 'MONITORING'
        self.kind = None
        self.deadline = None
        self.emit('Incident cancelled by wearer; status: MONITORING')
        return True

    def caregiver_pending(self):
        if self.state not in ('CONFIRMING', 'VERIFYING'):
            return False
        self.state = 'CAREGIVER_PENDING'
        self.deadline = None
        self.emit('CAREGIVER ALERT REQUIRED')
        return True

    def close_caregiver(self):
        """Finish an alert cycle so later sensor triggers can create a new incident."""
        if self.state != 'CAREGIVER_PENDING':
            return False
        self.state = 'MONITORING'
        self.kind = None
        self.deadline = None
        self.emit('Caregiver alert cycle finished; monitoring resumed')
        return True

    def tick(self):
        if self.state != 'CONFIRMING' or self.deadline is None:
            return
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            self.emit('No response received.')
            self.caregiver_pending()
