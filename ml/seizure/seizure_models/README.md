# Odyssey Guardian — simulated seizure-like motion classifiers

Trained on uploaded UEA Epilepsy dataset using its **original participant-disjoint** train/test split (3 participants train, 3 test). 275 recordings total, 206 timesteps × 3 wrist accelerometer axes at 16 Hz.

**Models:** `epilepsy_rf.joblib` (sklearn Random Forest, 44 statistical/spectral features computed by `train_epilepsy_models.py`), `epilepsy_cnn.pt` (PyTorch small 1D CNN). `evaluation.json` records held-out metrics. `test_samples.npz` contains examples from the test split for offline verification; do not train on it.

**Reproduce training:** `python train_epilepsy_models.py --zip Epilepsy.zip --output seizure_models` with numpy, scipy, scikit-learn, joblib, torch installed.

**Critical limitations:** All positive samples are staged/mimicked movements by healthy participants, not clinically verified epileptic seizures. The dataset contains only walking, running, and sawing as negative classes. It is NOT validated for real-world alerts. The exact unit scaling and sensor orientations need examination before integrating with the Nano 33 BLE Sense. The model expects **206 consecutive samples at 16 Hz** (12.875 seconds), not the 5-second windows proposed earlier. You cannot feed arbitrary 5-second data to the current model. For faster onset, retrain with shorter windows, verifying window labels, and assess false alarm behavior on real Arduino readings. Do not claim these scores measure medical diagnostic accuracy.

**Confusion matrix convention:** `[[normal correctly rejected, normal false positives], [mimicked seizure missed, mimicked seizure recognized]]`.
