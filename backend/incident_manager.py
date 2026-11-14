"""Non-blocking incident confirmation state machine."""
import time


class IncidentManager:
    def __init__(self, duration=15, clock=time.monotonic, emit=print):
        self.duration, self.clock, self.emit = duration, clock, emit
        self.state = 'MONITORING'
        self.deadline = None

    def candidate(self):
        if self.state not in ('MONITORING', 'CANCELLED'):
            return False
        self.state = 'CANDIDATE'
        self.emit('FALL DETECTED')
        self.state = 'CONFIRMING'
        self.deadline = self.clock() + self.duration
        self.emit('Asking for confirmation: Are you okay? Type c then Enter to cancel.')
        return True

    def cancel(self):
        if self.state != 'CONFIRMING':
            return False
        self.state = 'CANCELLED'
        self.emit('Incident cancelled by wearer')
        return True

    def tick(self):
        if self.state != 'CONFIRMING':
            return
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            self.state = 'CAREGIVER_PENDING'
            self.emit('No response received.')
            self.emit('CAREGIVER NOTIFICATION REQUIRED')
            return
