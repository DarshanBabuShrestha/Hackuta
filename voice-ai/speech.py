"""Reusable voice confirmation for the Odyssey Guardian fall workflow."""
from pathlib import Path
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import torch
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.speaker import SpeakerRecognition

SAMPLE_RATE = 16000
DURATION = 4
THRESHOLD = 0.45

class VoiceVerifier:
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or Path(__file__).parent / 'pretrained_models' / 'voice')
        self.model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=str(model_dir), local_strategy=LocalStrategy.COPY)
        self.recognizer = sr.Recognizer()
        self.owner_voice = None

    def record_voice(self, prompt, wait_for_enter=True, on_started=None, on_recorded=None):
        print(f'\n{prompt}')
        if wait_for_enter:
            input('Press Enter to begin recording...')
        print('Recording...')
        if on_started:
            on_started()
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        voice = audio.flatten()
        if on_recorded:
            on_recorded()
        if np.sqrt(np.mean(voice ** 2)) < 0.005:
            raise ValueError('No clear speech detected. Try again.')
        return voice

    def enroll(self):
        self.owner_voice = self.record_voice("Owner: say 'I'm okay'")

    def recognize_words(self, voice):
        pcm = (np.clip(voice, -1, 1) * 32767).astype(np.int16).tobytes()
        try:
            return self.recognizer.recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2)).lower()
        except (sr.UnknownValueError, sr.RequestError):
            return ''

    def verify_response(self, response):
        if self.owner_voice is None:
            raise RuntimeError('VoiceVerifier must be enrolled before verification')
        owner = torch.from_numpy(self.owner_voice).unsqueeze(0)
        reply = torch.from_numpy(response).unsqueeze(0)
        with torch.inference_mode():
            score, _ = self.model.verify_batch(owner, reply, threshold=THRESHOLD)
        score = float(score.item())
        text = self.recognize_words(response)
        said_okay = any(p in text for p in ("i'm okay", "i'm ok", 'i am okay', 'i am ok'))
        return score, score >= THRESHOLD, said_okay, text

    def confirm_fall(self, on_started=None, on_recorded=None):
        response = self.record_voice(
            "System: Say 'I'm okay' within 4 seconds.",
            wait_for_enter=False,
            on_started=on_started,
            on_recorded=on_recorded,
        )
        return self.verify_response(response)

if __name__ == '__main__':
    verifier = VoiceVerifier()
    print('=== VOICE REGISTRATION ===')
    verifier.enroll()
    print('Voice registered for this session.')
    print(verifier.confirm_fall())
