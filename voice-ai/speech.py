
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import torch
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.speaker import SpeakerRecognition

SAMPLE_RATE = 16000
DURATION = 4
THRESHOLD = 0.25  # SpeechBrain example default; NOT security-calibrated

print("Loading pretrained speaker verification model...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/voice",
    local_strategy=LocalStrategy.COPY
)

recognizer = sr.Recognizer()


def record_voice(message):
    print("\n" + message)
    input("Press Enter to begin recording...")

    print("Recording...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    voice = audio.flatten()

    if np.sqrt(np.mean(voice ** 2)) < 0.005:
        raise ValueError("No clear speech detected. Try again.")

    return voice


def recognize_words(voice):
    pcm = (np.clip(voice, -1, 1) * 32767).astype(
        np.int16
    ).tobytes()

    audio_data = sr.AudioData(pcm, SAMPLE_RATE, 2)

    try:
        return recognizer.recognize_google(audio_data).lower()
    except (sr.UnknownValueError, sr.RequestError):
        return ""


def verify_speaker(owner_voice, response_voice):
    owner = torch.from_numpy(owner_voice).unsqueeze(0)
    response = torch.from_numpy(response_voice).unsqueeze(0)

    with torch.inference_mode():
        score, prediction = model.verify_batch(
            owner,
            response,
            threshold=THRESHOLD
        )

    return (
        float(score.item()),
        bool(prediction.item())
    )


# STEP 1: Register the watch owner
print("\n=== VOICE REGISTRATION ===")

owner_voice = record_voice(
    "Owner: Say 'I'm okay,"
)

print("Voice registered for this session!")


# STEP 2: Simulate a fall
while True:
    print("\n=== FALL DETECTED ===")
    print("System: Are you okay? Please respond.")

    response = record_voice(
        "Say 'I'm okay, '"
    )

    # Identify the speaker
    score, is_owner = verify_speaker(
        owner_voice,
        response
    )

    # Understand the response
    spoken_text = recognize_words(response)

    print(f"\nRecognized speech: {spoken_text}")
    print(f"Speaker similarity score: {score:.3f}")

    said_okay = any(
        phrase in spoken_text
        for phrase in [
            "i'm okay",
            "i'm ok",
            "i am okay",
            "i am ok"
        ]
    )

    if is_owner and said_okay:
        print("Owner voice tentatively verified.")
        print("Request physical confirmation.")
        print("Emergency alert NOT automatically cancelled.")
    else:
        print("Owner confirmation NOT verified.")
        print("Keep emergency alert active.")
        print("Notify caregiver if confirmation times out.")

    again = input("\nTest another person? (y/n): ")
    if again.lower() != "y":
        break
