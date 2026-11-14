# Voice Verification for a Fall-Detection Smartwatch

A Python prototype that checks **what someone said** and **whether their voice resembles the enrolled watch owner's voice** after a simulated fall.

This is one component of a planned Arduino Nano 33 BLE Sense fall-detection and caregiver-alert system. **The current script runs on a computer using its microphone; it does not yet communicate with the Arduino, detect real falls, contact a caregiver, or dispatch emergency services.**

> **Safety notice:** This is a hackathon/research prototype, **not a medical device or a reliable emergency-response system**. A speaker match is not proof of identity or safety. Do not automatically cancel an emergency alert based only on voice recognition.

## Features

- Records an initial voice sample to enroll the wearer **for the current program session**.
- Records another sample when a fall is simulated.
- Uses a pretrained **SpeechBrain ECAPA-TDNN** model to calculate speaker similarity.
- Uses the `SpeechRecognition` library's Google recognizer to transcribe the response.
- Checks whether the response includes a phrase such as "I'm okay".
- Reports whether the response *tentatively* matches the enrolled voice and expected words.
- Keeps the emergency workflow unresolved until an additional confirmation step (not yet implemented).

**No custom speaker-model training is required.** The model is downloaded on first use. The current code does not permanently save the owner's voice between sessions.

## Technology

| Component | Purpose |
| --- | --- |
| Python 3.13 | Runtime used for development and testing |
| PyTorch / TorchAudio | Deep-learning inference and audio dependencies |
| SpeechBrain | Pretrained ECAPA-TDNN speaker verification |
| sounddevice | Record audio from the computer microphone |
| NumPy | Process audio samples |
| SpeechRecognition | Recognize spoken words |
| Hugging Face Hub | Download the pretrained model |

Model: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

## Project structure

```text
voice-ai/
├── speech.py           # Main voice-verification prototype
├── requirements.txt    # Python dependencies
├── README.md           # This guide
└── .gitignore          # Optional: ignore local/generated files
```

Generated directories such as `.venv/`, `pretrained_models/`, and `__pycache__/` should not be committed to source control.

## Prerequisites

- Windows, macOS, or Linux computer with a working microphone.
- Python **3.13** (used for this project's Windows testing); other versions may work but have not been verified here.
- Internet connection to download the speaker model on first run **and** to use `recognize_google()` for speech transcription.
- Enough disk space for PyTorch and the pretrained speaker model.

The Arduino board is **not** needed to test the current Python-only prototype.

## Installation — Windows PowerShell

1. Open a terminal inside the project folder. A short path helps avoid Windows path-length errors:

   ```powershell
   cd C:\voice-ai
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell prevents activation, run the same commands using `.\.venv\Scripts\python.exe` instead of `python`.

3. Update pip:

   ```powershell
   python -m pip install --upgrade pip
   ```

4. Install **CPU-only** PyTorch and TorchAudio (sufficient for this demo):

   ```powershell
   python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. Install the remaining project packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Verify the important imports:

   ```powershell
   python -c "import torch; print('PyTorch:', torch.__version__)"
   python -c "import speechbrain; print('SpeechBrain imported successfully')"
   ```

## Installation — macOS / Linux

```bash
cd voice-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For machine-specific PyTorch installation options, see the [official PyTorch installer](https://pytorch.org/get-started/locally/). Package availability varies by Python version, CPU architecture, and operating system.

## Important Windows fix for SpeechBrain

Windows may block the symbolic links that SpeechBrain normally uses to place model files in its local folder. If you see **`[WinError 1314] A required privilege is not held by the client`**, use the copy strategy in `speech.py`:

```python
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/voice",
    local_strategy=LocalStrategy.COPY,
)
```

This avoids needing Administrator privileges or Developer Mode for this operation. A Hugging Face cache symlink *warning* can still appear and is generally nonfatal.

## Running the demo

From the activated virtual environment:

```powershell
python speech.py
```

1. Allow microphone access if your operating system asks.
2. The program downloads/loads its pretrained model on first run.
3. Press **Enter** to register the watch owner's voice by reading the requested phrase.
4. The program simulates a fall and prompts for a second recording.
5. Say the requested response, then read the displayed transcription and similarity score.
6. Choose `y` to try another speaker, or `n` to exit.

The current demonstration records each clip for approximately **four seconds**, uses **16 kHz mono audio**, and enrolls only **one reference recording** for that run.

### Suggested tests

| Test | Speaker | Spoken response | What to observe |
| --- | --- | --- | --- |
| 1 | Registered owner | "I'm okay, please cancel the emergency alert." | Whether speaker and words are recognized |
| 2 | Different person | The **same sentence** | Whether the speaker check rejects the impostor |
| 3 | Registered owner | "Help me" | Whether the words prevent a false "I'm okay" result |
| 4 | Phone playback | Recording of the owner saying the expected sentence | Whether a replay can fool the model |
| 5 | Registered owner | Same phrase in a noisier room | How noise affects similarity and transcription |

**Scores are similarity measurements, not probabilities of identity.** The example's `0.25` speaker threshold is only a starting point and has **not** been calibrated for this project. Collect multiple samples from several speakers, measure false acceptances and false rejections, and select a threshold based on those results. Even a calibrated threshold cannot guarantee identity.

## System behavior and safety

The demo evaluates two distinct questions:

1. **Speech recognition:** Did the recorded words indicate that the speaker said "I'm okay"?
2. **Speaker verification:** Does the voice resemble the enrolled owner's voice?

A positive result from both tests should be treated as **tentative**, not as authorization to cancel an emergency. Real systems also need physical confirmation, a timeout, a caregiver escalation path, connectivity monitoring, and protection against replayed voices. If the user does not respond, recognition fails, or a different speaker responds, the emergency workflow should **not silently cancel**.

The sample program **prints** messages about notifications; it does not actually send notifications or request emergency dispatch.

### Privacy

- Microphone recordings contain potentially sensitive biometric information.
- The sample's `recognize_google()` transcribes audio using an external service; it is **not an offline-only solution**.
- Obtain consent before recording others, limit retention, and avoid uploading raw voice recordings to public repositories.
- Consider local/offline speech recognition before moving beyond a demo.

## Troubleshooting

**`WinError 1314` when loading the model**  
Use `LocalStrategy.COPY` as shown above.

**Long-path error while installing `torch` on Windows**  
Use a short location such as `C:\voice-ai` with a virtual environment. Windows long-path support may also need to be enabled.

**`ModuleNotFoundError` after installing packages**  
Confirm the virtual environment is active and check that VS Code is using its interpreter (`.venv\Scripts\python.exe` on Windows).

**No microphone / silent recording**  
Check OS microphone permission, select the correct input device, and verify that the microphone works in another application.

**Transcription fails or returns empty text**  
Check internet access, speak clearly, and try a quieter room. A failed transcript should not be treated as a successful confirmation.

**Incorrect speaker matches**  
Record in a quieter room, avoid very short or silent samples, collect more enrollment examples, and evaluate the threshold against multiple other speakers. The pretrained model is not spoof-proof.

## Planned Arduino integration

The intended system is:

```text
Arduino Nano 33 BLE Sense
       │
       ├── Motion sensors → TinyML fall detection
       │
       └── Fall event → Computer / companion app
                           │
                           ├── Prompt wearer for a response
                           ├── Transcribe response
                           ├── Verify enrolled speaker
                           ├── Request additional confirmation
                           └── Notify caregiver if confirmation fails/times out
```

**Not yet implemented:** Arduino-to-computer communication, live fall detection in this application, persistent voice enrollment, physical confirmation, caregiver messaging, location sharing, and any dispatch integration.

## Dependencies and reproducibility

`requirements.txt` lists the direct dependencies. For a team reproducibility snapshot, create a version-pinned file **after** confirming the environment works:

```powershell
python -m pip freeze > requirements-lock.txt
```

A lock generated on one operating system may not install unchanged on another, especially for PyTorch. Teammates should use a compatible PyTorch build and the documented Python version.

## Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for pretrained speaker verification.
- [Hugging Face](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) for hosting the speaker model.
- [PyTorch](https://pytorch.org/) for the machine-learning runtime.
