# Odyssey Guardian experimental fall detection MVP

This prototype streams six-axis motion from an original Arduino Nano 33 BLE Sense to a Windows laptop. Python classifies each 32-reading window for falls and separately reports an experimental simulated-seizure-like movement score after collecting a 206-sample acceleration window. Possible falls and seizure-like motion start a 15-second confirmation timer. An unconfirmed incident appears in the caregiver page; if the caregiver does not acknowledge it within 10 seconds, a configured Twilio account requests a voice call.

## What is in this workspace

The baseline training script and model became available during implementation. The original sensor sketch was located outside this project and copied into `arduino/`. The baseline results JSON was empty, and the ZIP was absent; the extracted `ml/full_dataset/Dataset` CSVs were present. The baseline model loads and predicts, but it was trained on complete recordings. `ml/train_odyssey_windows.py` applies its **exact 56-feature extractor** to fixed windows and saves a distinct model at `ml/odyssey_window_model.joblib`; it does not replace the baseline.

## Windows installation

Install Python 3.10+ and Arduino IDE. In PowerShell from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The supplied `odyssey_window_model.joblib` can be used directly. To retrain from the extracted CSVs, run `python -m ml.train_odyssey_windows`. Requirements pin scikit-learn 1.6.1 to match this window model. The separate baseline model was saved with scikit-learn 1.8; loading it in the local 1.6.1 environment produced a version warning, although a test prediction succeeded. Only load model files you trust.

## Arduino upload and port

In Arduino IDE, install **Arduino_LSM9DS1** through Library Manager. Select **Arduino Nano 33 BLE** under Tools → Board, open `arduino/OdysseyGuardian_FallDetection/OdysseyGuardian_FallDetection.ino`, select the board's port under Tools → Port, and upload. The separate `OdysseyGuardian_SensorTest.ino` prints a different format and cannot drive the dashboard. If the port is unclear, unplug the USB cable and note which COM port disappears in Windows Device Manager → Ports (COM & LPT); reconnect and use that port. Close Arduino Serial Monitor before running Python.

The sketch sends `IMU,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps` at a scheduled 50 Hz. A line with a missing sensor pair is skipped. Python ignores the header and shows firmware errors in the terminal and dashboard. If the IMU fails initialization, the sketch repeats an error and stops sampling.
If the backend opens COM9 but receives no samples, stop the backend, upload the fall-detection sketch above, close Arduino Serial Monitor, and restart the backend. The sketch now repeats `ERROR,IMU_INIT_FAILED` if sensor initialization fails and sends `STATUS,WAITING_FOR_IMU` if it is running without readings; the dashboard displays these states.

## Run live

```powershell
python -m backend.main --port COM9
```

Use your actual port. A live serial run starts the local dashboard and prepares voice enrollment automatically. Open `http://127.0.0.1:8000` and click **Record my voice** when the voice model is ready. Optional flags: `--baud 115200`, `--no-web` for terminal-only mode. Type `c` then Enter while the timer runs to cancel. Python continuously reads serial on a worker thread, so the timer continues during sensor collection or disconnection. Exit with Ctrl+C.

The live dashboard at `http://127.0.0.1:8000` shows motion and detector updates over a WebSocket, plus the incident state and recent events. The **I'm okay** and **Need help** buttons work during confirmation. Use `--web-port 8001` to change the port. The wearer dashboard is available only on this computer. Add `--web` explicitly for a simulation run.
The default `--mode all` enables fall, tremor-like, and seizure-like motion. `--mode fall` turns the other two off. The seizure score first appears after about 13 seconds of continuous samples; a normal-activity score is still a valid result.

```powershell
python -m backend.main --port COM9 --web
python -m backend.main --simulate ml/full_dataset/Dataset/fall/user1/user1_fall2.csv --web
```

## Caregiver alerts and phone escalation

Start the backend with `--web-host 0.0.0.0` to let a caregiver's phone on the same Wi-Fi network open `http://<laptop-LAN-IP>:8000/caregiver`. Find the laptop's LAN IP with `ipconfig` on Windows. The backend prints a caregiver access key; enter it on the phone and keep the page open. Set `ODYSSEY_CAREGIVER_KEY` before starting if you want a stable key. The caregiver page shows the incident type and time, plays a short sound where the browser permits it, and offers **Acknowledge alert**. Browser notifications require a secure context on most phones, so the visual alert is the dependable part of this local HTTP demo. The wearer dashboard shows whether a caregiver page is connected.

For phone escalation, set these environment variables before starting the backend. Use E.164 phone numbers, including `+` and country code:

```powershell
$env:TWILIO_ACCOUNT_SID = "AC..."
$env:TWILIO_AUTH_TOKEN = "..."
$env:TWILIO_FROM_NUMBER = "+15551234567"
$env:ODYSSEY_CAREGIVER_PHONE = "+15557654321"
$env:TWILIO_TWIML_URL = "https://handler.twilio.com/twiml/..."
python -m backend.main --port COM9 --web-host 0.0.0.0
```

Twilio trial accounts can call only verified destination numbers. Verify
`ODYSSEY_CAREGIVER_PHONE` in the Twilio console and use the Twilio-provided trial
number for `TWILIO_FROM_NUMBER`. Create a TwiML Bin in the Twilio console with:

```xml
<Response>
  <Say>Odyssey Guardian alert. The wearer may need immediate assistance. Please check on them now.</Say>
</Response>
```

Set `TWILIO_TWIML_URL` to that Bin's public URL. Trial projects can reject the
inline `Twiml` call parameter, so the backend sends the supported `Url` parameter
instead. If `TWILIO_TWIML_URL` is omitted, it uses Twilio's public sample voice
URL; this confirms calling works but plays Twilio's sample message rather than
the Guardian alert.

For a simulation with `--web`, the replay waits until you click **Start replay** on the wearer dashboard. Wait for **Ready to enroll**, click **Record my voice**, say “I'm okay”, and then start the replay. You can start the replay without voice enrollment if you want to use the manual buttons. This avoids a replayed fall blocking voice enrollment before the model is ready.

After wearer confirmation expires or **Need help** is pressed, the caregiver page receives a new alert. An acknowledgment received before 10 seconds stops escalation. Otherwise the backend requests one outbound Twilio voice call. After acknowledgment or a call result, the incident returns to monitoring so a later fall can open a new confirmation and caregiver alert. The UI distinguishes a call request accepted by Twilio from a call actually answered; this prototype does not track answer status. If credentials are missing or Twilio rejects the request, the UI displays the failure. The caregiver page must stay open and connected for app alerts; this prototype does not include background mobile push. Keep this demo on a trusted local network. For internet access, put it behind HTTPS and add production authentication and durable incident storage before use. [Twilio's Call API](https://www.twilio.com/docs/voice/api/call-resource) documents the outbound request and call charges.

## Safe controlled demo

Secure the board and cable, keep the laptop stable, and move the board by hand over a soft surface. Try ordinary movements first, then a controlled quick change of orientation and a gentle tap on a cushion. Do not ask a person to fall. The model may not trigger from a staged gesture; collect and inspect the model scores to understand missed detections.

## Simulation and tests

Replay a BITS-2 recording (no Arduino required):

```powershell
python -m backend.main --simulate ml/full_dataset/Dataset/fall/user1/user1_fall2.csv
python -m backend.main --simulate ml/full_dataset/Dataset/adl/user1/user1_adl1.csv --speed 4
python -m pytest -q
```

An Arduino-format CSV can also be replayed using `--simulate path\to\capture.csv`. For a fast classifier test, use `--speed 0`; if an incident triggers, the confirmation countdown still takes 15 real seconds. The replay synthesizes 20 ms timestamps from sample position.

## Model and sensor limitations

BITS-2 contains separate gravity-included acceleration (`acg`) and gyroscope (`gyro`) rows. Their typical magnitudes suggest m/s² and rad/s, respectively, but the baseline artifact records only "raw dataset units preserved" and provides no explicit unit metadata. The CSV timestamps are rounded too heavily to align these streams precisely. Training aligns each recording's gyro sequence to its acceleration sequence by relative position, then takes non-overlapping 32-sample windows. Live Arduino values are converted from g to m/s² and degrees/s to radians/s. The feature vector calls the original baseline script: for `acg` then `gyro`, each x, y, z, and magnitude has mean, standard deviation, 5th percentile, median, 95th percentile, maximum absolute value, and mean absolute sample-to-sample change. The artifact checks this exact feature contract before inference.

Training, validation, and test are separated by participant. Fall labels are inherited from entire recordings, so some fall windows may show pre- or post-fall motion. Different sensor placement, range, filtering, sample timing, and synchronization between BITS-2 and the Arduino can cause substantial accuracy loss. The 32 samples cover about 0.64 seconds at the sketch's nominal 50 Hz; missed pairs lengthen that interval. These conditions mean offline test metrics are **not** live sensitivity, false-alarm rate, or clinical validation. This is an experimental hackathon prototype, not a certified safety device.

The held-out participant test split for this window model reached 79.7% accuracy and 61.5% fall recall (`ml/odyssey_window_results.json`). The original recording-level report states 92.36% accuracy and 83.33% fall recall; these are different tasks and should not be compared as if they measure live detection. Arduino acceleration and gyro reads are sequential rather than hardware synchronized. Live sampling and firmware compilation still require verification on the board.

## Voice confirmation integration

With `--web`, the backend prepares voice confirmation automatically. Open the dashboard, click **Record my voice**, and say “I'm okay” into the computer microphone during the four-second recording. No terminal input is needed. When a possible fall or seizure-like movement occurs, the computer speaks “Are you okay? Please confirm” once. After enrollment, it then records a four-second response and checks the enrolled voice and words. A matching response tentatively cancels the incident. If the voice is not recognized, the computer speaks “Voice not recognized. Alerting the caregiver.” and opens the caregiver alert. The dashboard shows the selected microphone and each step. `--voice-confirmation` is still accepted as an alias that opens the dashboard.

```powershell
python -m backend.main --port COM9 --web
```

This is an experimental hackathon feature, not secure identity verification or a medical safety guarantee. Speech transcription uses Google's recognition service.

## Experimental seizure-like movement score

The backend loads `ml/seizure/seizure_models/epilepsy_rf.joblib`, buffers the acceleration axes from the live 50 Hz stream, selects samples at the artifact's nominal 16 Hz cadence, and calls the exact `features()` function in `ml/seizure/train_epilepsy_models.py`. A concurrent rolling five-second FFT checks rhythmic power, while each one-second portion must have measurable acceleration variability. A possible seizure-like movement incident requires that motion check, a filled 206-sample ML window (about 12.9 seconds), and a seizure-class probability above the configured threshold. There is no additional five-second wait after the ML window fills. The dashboard and console show motion status, rhythmic duration, probability, and the suppression reason. The fall detector uses a configurable 0.7 probability threshold and 10-second trigger cooldown (`--threshold` and `--cooldown`). The seizure thresholds are experimental defaults, not calibrated to this Arduino: `--seizure-motion-rms-g 0.12`, `--seizure-peak-power-share 0.08`, `--seizure-band-power-share 0.50`, `--seizure-low-hz 3`, `--seizure-high-hz 8`, `--seizure-rhythmic-seconds 5`, and `--seizure-threshold 0.35`. Measure stationary noise and representative wearer movement before tuning them. Model training used staged movements, and Arduino units, orientation, placement, and preprocessing have not been fully calibrated to the dataset. This output is possible seizure-like movement, not a confirmed epileptic seizure or a medical diagnosis.
When a model first changes to its positive class, the backend starts the same confirmation flow. If both models first predict positive in the same window, the incident identifies both. Recent events shows the detection and confirmation steps. Another incident can start after the relevant model predicts negative and later positive again.

## Run detector modes

`--mode all` runs fall, tremor-like motion, and simulated seizure-like motion (the default). Select a single mode to test only that detector. Voice confirmation applies to fall, seizure, and all modes.

```powershell
py -m backend.main --port COM9 --mode all --voice-confirmation
py -m backend.main --port COM9 --mode fall --voice-confirmation
py -m backend.main --port COM9 --mode tremor
py -m backend.main --port COM9 --mode seizure
```

The tremor-like detector is a five-second spectral motion heuristic, not a clinically validated tremor sensor. `--mode epilepsy` is also accepted as an alias for `--mode seizure`.

