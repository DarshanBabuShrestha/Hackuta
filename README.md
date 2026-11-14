# Odyssey Guardian experimental fall detection MVP

This prototype streams six-axis motion from an original Arduino Nano 33 BLE Sense to a Windows laptop. Python classifies each 32-reading window for falls and separately reports an experimental simulated-seizure-like movement score after collecting a 206-sample acceleration window. Possible falls start a 15-second confirmation timer. It prints a caregiver alert requirement only; it does not contact anyone.

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

In Arduino IDE, install **Arduino_LSM9DS1** through Library Manager. Select **Arduino Nano 33 BLE** under Tools → Board, open `arduino/OdysseyGuardian_FallDetection.ino`, select the board's port under Tools → Port, and upload. If the port is unclear, unplug the USB cable and note which COM port disappears in Windows Device Manager → Ports (COM & LPT); reconnect and use that port. Close Arduino Serial Monitor before running Python.

The sketch sends `IMU,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps` at a scheduled 50 Hz. A line with a missing sensor pair is skipped. The header and error lines are ignored by Python. If the IMU fails initialization, the sketch reports an error and stops sampling.

## Run live

```powershell
python -m backend.main --port COM4
```

Use your actual port. Optional flags: `--baud 115200`, `--threshold 0.7`, `--cooldown 20`. Type `c` then Enter while the timer runs to cancel. Python continuously reads serial on a worker thread, so the timer continues during sensor collection or disconnection. Exit with Ctrl+C.

## Safe controlled demo

Secure the board and cable, keep the laptop stable, and move the board by hand over a soft surface. Try ordinary movements first, then a controlled quick change of orientation and a gentle tap on a cushion. Do not ask a person to fall. The model may not trigger from a staged gesture; `--threshold` can be lowered for a software demo, but that does not improve real-world accuracy.

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

## Experimental seizure-like movement score

The backend loads `ml/seizure/seizure_models/epilepsy_rf.joblib`, buffers the acceleration axes from the live 50 Hz stream, resamples to the artifact's 16 Hz cadence, and calls the exact `features()` function in `ml/seizure/train_epilepsy_models.py`. It reports a probability and class after the 206-sample (about 12.9-second) buffer fills. `--seizure-threshold` changes the displayed class cutoff (default `0.5`). This score is experimental only: model training used staged movements, and Arduino units, orientation, placement, and preprocessing have not been calibrated to the dataset. It must not be treated as seizure detection or used for medical decisions.
