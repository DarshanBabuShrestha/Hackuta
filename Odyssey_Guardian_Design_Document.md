Version 1.0 • 14 November 2026 • Build target: 24–36-hour hackathon

# 1. Product overview

Odyssey Guardian is a wrist-worn safety-monitor concept that uses an Arduino Nano 33 BLE Sense to recognize possible falls and tremor-like motion, requests confirmation from the wearer, and alerts a caregiver when assistance may be needed. A laptop-connected MVP provides the audible prompt, event processing, and caregiver dashboard. The design includes an experimental respiratory-sound classifier and a simulated dispatch escalation for the demo.

## The problem

People who experience falls or movement-related incidents may be unable to ask for help promptly. Traditional SOS devices depend on a manual press; Odyssey Guardian combines automatic pattern detection with an explicit opportunity to cancel false alarms.

## Design decisions at a glance

| **Decision**       | **MVP approach**                                                                                                                 |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Sensors            | Built-in accelerometer, gyroscope, microphone; no purchased sensors.                                                             |
| SOS input          | Programmable button on the Tiny Machine Learning Shield (D13); verify electrical behavior.                                       |
| Alerts and sound   | Laptop speaker, USB serial connection, and a caregiver web dashboard.                                                            |
| Location           | Opt-in location from a companion phone/browser; otherwise display “Location unavailable” or an explicitly labeled demo location. |
| Emergency response | Simulated dispatch for the hackathon; never claim a real 911 call or real dispatch.                                              |
| Budget             | Additional hardware cost: \$0 for the tethered MVP, assuming laptop/phone access.                                                |

## Success criteria

- Watch-like hardware identifies a staged fall, starts a countdown, and accepts a voice or button response.

- A caregiver can view incident type, time, confidence/trigger details, location status, and acknowledge or decline.

- An unacknowledged event enters a clearly labeled simulated escalation; every status change is logged.

- Demo works without the camera, a buzzer, a vibration motor, a breadboard, or a separate SOS button.

## Explicit non-goals

No medical diagnosis, seizure prediction, certified fall detection, continuous clinical wheeze monitoring, independent GPS, guaranteed network delivery, or automatic live emergency-services dispatch. The laptop-connected MVP is not an autonomous consumer smartwatch.

## Reference sources

\[1\] Arduino Tiny Machine Learning Kit (Arduino). \[2\] Nano 33 BLE Sense sensors (Arduino). \[3\] TinyML Kit Hardware Workshop (Harvard/UNIFEI). \[4\] Calling 911 (911.gov).

# 2. Hardware inventory and wiring

The kit’s original Nano 33 BLE Sense has an IMU, digital microphone, Bluetooth Low Energy, and USB connectivity. Arduino notes that some kit batches omit the humidity/temperature sensor; that omission does not affect this design \[1–3\].

| **Component**                | **Owned?**        | **Use in MVP**                                                                                                       |
|------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------|
| Nano 33 BLE Sense            | Yes               | Motion data (accelerometer + gyroscope), microphone experimentation, event logic; model inference where feasible.    |
| Tiny Machine Learning Shield | Yes               | Programmable push-button; physical mounting and expansion connectors.                                                |
| OV7675 camera                | Yes               | Not used in MVP: wrist camera adds little value for fall/tremor detection and introduces privacy/compute challenges. |
| USB cable                    | Yes               | Power, firmware upload, and serial event stream to laptop.                                                           |
| Laptop with speaker          | Available to team | Voice prompts, local application, backend, alarms, demo console.                                                     |
| Phone (optional)             | Team-owned        | Location with explicit permission; caregiver dashboard for mobile demonstration.                                     |

## Button mapping

| **Input**                                      | **Action**  | **Notes**                                                         |
|------------------------------------------------|-------------|-------------------------------------------------------------------|
| Shield button: hold ≥2 s                       | Manual SOS  | Debounce; hold threshold is proposed behavior and must be tested. |
| Shield button: short press during confirmation | “I am okay” | Outside the confirmation state a short press does nothing.        |
| Nano onboard reset button                      | Reset only  | Do not map this button to SOS; pressing it restarts the device.   |

Electrical note: a hardware workshop reference maps the shield’s push button to D13 \[3\]. D13 also serves the Nano’s built-in LED/SCK function, so avoid simultaneous LED/SPI assumptions and verify button polarity with a short test sketch before writing the full firmware.

## What is and is not available

- Already available: motion sensing, movement-frequency analysis, audio input, USB serial, onboard status LED, and a shield push button.

- Not built in: an actual speaker, vibration actuator, GPS/GNSS receiver, cellular modem, battery charger, or guaranteed emergency-service connection.

- For version 1, laptop audio replaces the buzzer; accelerometer detects shaking but cannot physically vibrate the watch. The USB cable replaces a battery.

## Future wearable hardware, only after the demo

Add a protected battery and correctly matched charger/regulator, compact enclosure/strap, optional haptic motor, and BLE link to a phone. Follow 3.3 V GPIO rules and safe power-design practices. These additions are intentionally outside the \$0 MVP.

# 3. End-to-end architecture

The USB MVP keeps sensing close to the device and puts networking, audible prompts, and caregiver interactions on the computer. This is the lowest-risk architecture for a short hackathon.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>WEARABLE: Arduino + TinyML Shield</strong></p>
<p>IMU windows • button • optional microphone inference</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

↓ USB serial: sensor/event messages

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>LOCAL BRIDGE: Python on laptop</strong></p>
<p>Event validation • countdown • prompts • incident state machine</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

↓ HTTP/WebSocket over local or hosted network

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>BACKEND + EVENT STORE</strong></p>
<p>Create/update incidents • timeout logic • audit history</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

↓ Live caregiver update

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>CAREGIVER WEB DASHBOARD</strong></p>
<p>Incident card • location/map link • Acknowledge / Cannot assist</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

↓ If unacknowledged: demo-only escalation

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>SIMULATED DISPATCH CONSOLE</strong></p>
<p>Shows request received and a manually simulated status; never implies a real responder</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Component responsibilities

| **Module**          | **Inputs**                              | **Outputs**                                                         |
|---------------------|-----------------------------------------|---------------------------------------------------------------------|
| Arduino firmware    | IMU, shield button, optional microphone | Timestamped sensor summaries and candidate incident events.         |
| Local Python bridge | USB serial events                       | Deduplicated events, speech/alarm playback, confirmation decisions. |
| Backend API         | Bridge events; caregiver responses      | Persistent incidents, status updates, dashboard stream.             |
| Caregiver dashboard | Incidents and location                  | Acknowledged / cannot assist; maps and event history.               |
| Location companion  | Opt-in browser/phone GPS                | Coordinates with accuracy and timestamp, or unavailable.            |
| Demo dispatcher     | Escalated test incidents                | Simulated “received” and “assigned” status only.                    |

## Connectivity rule

If USB or the network disconnects, show “Connection lost” and preserve unsent events locally where possible. Never treat an offline caregiver as having received an alert. Phone/browser location is optional and may not be precise indoors.

# 4. Incident and confirmation workflow

Use one state machine for both manual and automatically detected incidents. Timeouts below are demo defaults, not medically validated thresholds. Every incident needs a unique ID so duplicate sensor triggers do not produce repeated alerts.

| **State**                | **Trigger or decision**                  | **User-facing behavior**                                                                   |
|--------------------------|------------------------------------------|--------------------------------------------------------------------------------------------|
| MONITORING               | Normal activity                          | Quiet dashboard; monitor device connection.                                                |
| CANDIDATE                | Fall/tremor pattern or long SOS press    | Create incident, record trigger, timestamp.                                                |
| CONFIRMING (15 s)        | Laptop speaks: “Are you okay?”           | Display countdown; button short press or “I’m okay” cancels; “Help” skips waiting.         |
| CANCELLED                | User confirms okay                       | Show “Cancelled by wearer”; save test log, send no urgent caregiver alert.                 |
| CAREGIVER_PENDING (30 s) | Help requested or confirmation timeout   | Send caregiver incident and location if available; dashboard visibly rings.                |
| ACKNOWLEDGED             | Caregiver accepts                        | Show “Caregiver acknowledged”; do not claim they or responders arrived.                    |
| ESCALATION_DEMO          | Caregiver declines or no response        | Move to a simulated emergency queue with label “Demo — no real call placed”.               |
| SIMULATED_DISPATCH       | Demo operator clicks “Received/Assigned” | Only show “Simulated dispatch confirmed”; real dispatch requires trusted outside evidence. |
| CLOSED                   | Incident resolved in demo                | Store final state, timing, and response path.                                              |

## Interaction rules

- Manual SOS overrides detection confidence: a ≥2-second button hold goes directly to caregiver alert after a brief local acknowledgement.

- “I’m okay” / short press cancels only while the confirmation prompt is active; log how the cancellation happened.

- If speech recognition is uncertain, retain the countdown and allow the physical button as the reliable fallback.

- A caregiver acknowledgment confirms notification handling, not arrival of help. A caregiver “cannot assist” response triggers demo escalation.

- For real emergencies, direct the wearer/caregiver to call 911 and provide event details and location. Do not program a live 911 test call \[4\].

## User-visible status labels

Use precisely distinct wording: “Possible fall detected” → “Checking wearer” → “Alert sent to caregiver” → “Caregiver acknowledged” or “No caregiver response” → “Simulated escalation queued” → “Simulated dispatch received.” Never display “Police dispatched” unless verified by an actual authorized responder system.

# 5. Detection and TinyML design

Prioritize reliably demonstrating two motion features before adding audio classification. Begin with interpretable baselines; add a TinyML model only after data collection and validation.

## A. Fall-like event detection — MVP / required

> 1\. Sample acceleration (ax, ay, az) and angular velocity (gx, gy, gz) from the IMU. Use the sensor’s available sample rate and a rolling window.
>
> 2\. Calculate acceleration magnitude: \|a\| = sqrt(ax² + ay² + az²). Flag a candidate when a large acceleration peak and a marked orientation change or unusual inactivity occur close together.
>
> 3\. Start a 15-second confirmation window. Suppress repeated triggers for the same incident until it closes.
>
> 4\. Compare a transparent threshold baseline against a small 1-D CNN/TinyML classifier trained on labeled safe movement recordings.

A magnitude threshold around 2 g is only a starting experiment—not a production cutoff. Test against sitting, picking up the device, rapid hand motion, and ordinary walking. Do not deliberately fall while wearing the prototype.

## B. Tremor-like oscillation — MVP / required

> 1\. Capture 3–5-second accelerometer windows; remove gravity/slow drift and estimate the dominant oscillation frequency using an FFT or equivalent spectral method.
>
> 2\. Require sustained rhythmic movement and minimum motion amplitude; reject windows dominated by broad, irregular hand gestures.
>
> 3\. Display “Repetitive movement detected” with the estimated frequency and confidence. Treat this as a movement pattern, not a Parkinson’s or epilepsy diagnosis.

## C. Voice and sound — phased

| **Feature**        | **Implementation choice**                                                                                         | **Reality check**                                                                           |
|--------------------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Confirmation voice | Laptop microphone and speech recognizer for MVP; fixed keywords: “I’m okay” and “Help”.                           | Fastest reliable demo; on-device microphone inference is a stretch goal.                    |
| Wheezing           | Experimental classifier using Arduino PDM microphone or laptop audio; compare spectral features to ambient noise. | Wrist location is unreliable for breathing sounds; do not interpret as an asthma diagnosis. |
| Audio prompt       | Laptop text-to-speech or prerecorded WAV.                                                                         | Arduino has a microphone, not a speaker.                                                    |

## Dataset and evaluation protocol

Record labeled, consented examples from several volunteers: normal walking, sitting, vigorous arm gestures, controlled device-to-cushion impacts (not human falls), deliberate rhythmic wrist shaking, silence, speech, and background noise. Split evaluation by person, not randomly by nearly identical time windows. Report confusion matrices, missed staged incidents, false-alarm rate, and detection latency. Explain clearly that simulated gestures do not establish clinical accuracy.

# 6. Software design and event contract

| **Layer**           | **Proposed stack**                                                                            | **Responsibilities**                                                          |
|---------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Device firmware     | Arduino IDE/C++; Arduino_LSM9DS1 for original Nano, or correct Rev2 IMU library; PDM optional | Sample motion, debounce button, candidate detection, compact serial messages. |
| Local application   | Python; pyserial; optional text-to-speech and microphone recognizer                           | Read USB, run countdown, voice prompt, deduplicate, forward events.           |
| Server              | FastAPI + SQLite; WebSocket for live updates                                                  | Incident state transitions, caregiver actions, simple persistence.            |
| Caregiver dashboard | React or simple HTML/JS                                                                       | Incoming incident cards, map link, status, acknowledge/decline.               |
| Location sender     | Companion phone web page, with user permission                                                | Latitude/longitude/accuracy/timestamp when available; otherwise null.         |
| Emergency demo      | Backend simulated queue and operator toggle                                                   | Demonstrates escalation without making emergency calls.                       |

## Example incident payload

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>{<br />
"event_id": "demo-00042",<br />
"timestamp_utc": "2026-11-14T21:00:00Z",<br />
"event_type": "POSSIBLE_FALL",<br />
"source": "arduino_imu",<br />
"confidence": null,<br />
"confirmation": "NO_RESPONSE",<br />
"status": "CAREGIVER_PENDING",<br />
"location": {"lat": null, "lon": null, "source": "unavailable"},<br />
"is_demo": true<br />
}</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Minimal API routes

| **Route**                      | **Action**                                              |
|--------------------------------|---------------------------------------------------------|
| POST /api/events               | Bridge creates an incident.                             |
| GET /api/events                | Caregiver reads current incidents/history.              |
| POST /api/events/{id}/response | Caregiver submits ACKNOWLEDGED or CANNOT_ASSIST.        |
| POST /api/location             | Permissioned phone location update.                     |
| POST /api/demo-dispatch/{id}   | Demo-only operator updates simulated dispatcher status. |
| WS /ws/events                  | Dashboard receives real-time event updates.             |

## Data and privacy boundaries

Require caregiver sign-in for any externally hosted dashboard, use HTTPS, avoid storing raw microphone audio by default, minimize retention of exact location, and never expose incident links publicly. Use fake identities and simulated incidents for the hackathon. If no GPS permission is granted, display “Location unavailable”; a hard-coded demo pin must be labeled “Simulated location”.

# 7. Build plan, verification, and demo

## Recommended build order (24–36 hours)

| **Priority** | **Task**                                           | **Deliverable**                                                                     |
|--------------|----------------------------------------------------|-------------------------------------------------------------------------------------|
| P0 — first   | Verify shield button D13; stream IMU data over USB | Serial stream, SOS input, sensor diagnostics.                                       |
| P0           | Fall baseline + cooldown + 15-second prompt        | Staged fall-to-confirmation end-to-end path.                                        |
| P0           | FastAPI event store and caregiver UI               | Incident cards, acknowledgement, timeout handling.                                  |
| P0           | Location input + demo dispatch queue               | Permissioned phone location or clearly labeled test location; simulated escalation. |
| P1           | Tremor frequency pipeline                          | Repetitive movement detection with spectral view.                                   |
| P1           | Voice recognition and cancellation                 | “I’m okay” / “Help” through laptop microphone.                                      |
| P2           | TinyML classifier and experimental wheeze demo     | Evaluated against a baseline; never represented as medical-grade.                   |

## Acceptance test checklist

| **Test**             | **Pass condition**                                                                           |
|----------------------|----------------------------------------------------------------------------------------------|
| Manual SOS           | Long press creates one incident; short press does not accidentally trigger SOS.              |
| Staged possible fall | Safe board-to-cushion scenario causes prompt without physical human fall.                    |
| Cancellation         | Button or recognized “I’m okay” closes incident and does not alert caregiver.                |
| No response          | Timeout raises caregiver incident with timestamp and correct location availability label.    |
| Caregiver feedback   | Ack/decline updates immediately on wearer and caregiver screens.                             |
| Escalation demo      | Timeout/decline shows “Simulated” in every dispatch status; no real emergency call.          |
| Offline behavior     | USB/network loss is surfaced; no false “delivered” status.                                   |
| False-positive trial | Rapid arm gestures, sitting, lifting objects and ordinary movements are logged and reviewed. |

## 90-second judge demonstration

Start with the hardware and \$0 extra-hardware claim. Stage a board-to-cushion fall, let the 15-second confirmation expire, and show the incident arrive on the caregiver dashboard with a permissioned or clearly simulated location. Decline or allow caregiver timeout; demonstrate the SIMULATED dispatch screen. Close with short-press cancellation and a rhythmic-motion example.

## Known limitations and deployment gate

This is not a medical or emergency-response product. It may miss real events, flag ordinary activity, and fail when USB or internet disconnects. Wheeze capture is particularly unreliable at the wrist. Real deployment requires clinical testing, reliable battery/network design, security/privacy review, and an authorized emergency-response integration.
