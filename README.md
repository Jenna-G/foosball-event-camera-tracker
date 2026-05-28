# Foosball Vision Tracker

**Low-Latency Event Camera Demonstrator: Neuromorphic Robotic Foosball Goalkeeper System**

Real-time foosball ball tracking and goalkeeper position prediction using a **Prophesee EVK4 neuromorphic event camera** and a **DJI Osmo Action 4 frame-based camera**. Implements equivalent centroid-based detection and trajectory prediction algorithms across both platforms to enable direct hardware comparison.

Developed as part of a QUT Engineering capstone research project investigating event-based vs frame-based vision for high-speed object tracking.

---

## Research Context

**Research question:** How does event-based vision compare to frame-based approaches in detection latency, tracking accuracy, and maximum trackable velocities under visual occlusions and high-speed scenarios?

The core methodological contribution is **algorithmic equivalence** — both cameras run identical tracking and prediction pipelines, isolating hardware sensing differences from software optimisation. Where preprocessing differs (MOG2 background subtraction on the DJI), this is explicitly justified by hardware architecture rather than algorithmic advantage.

---

## Repository Structure

```
foosball-event-camera-tracker/
│
├── README.md
├── LICENSE
│
├── tracking_only/              # Standalone tracking scripts — no hardware actuator
│   ├── evk4/
│   │   ├── evk4_tracker.py         # Main EVK4 tracking + CSV logging
│   │   ├── calibrate_evk4.py       # Goal post calibration tool
│   │   └── led_calibrate_evk4.py   # LED sync ROI calibration tool
│   └── dji/
│       ├── dji_tracker.py          # Main DJI tracking + CSV logging
│       ├── calibrate_dji.py        # Goal post + ROI calibration tool
│       └── led_calibrate_dji.py    # LED sync ROI calibration tool
│
└── goalkeeper_integration/     # Tracking integrated with goalkeeper rod actuator
    ├── evk4/
    │   └── evk4_goalkeeper.py      # EVK4 tracker with Arduino serial output
    └── dji/
        └── dji_goalkeeper.py       # DJI tracker with Arduino serial output
```

---

## Hardware

| Component | Specification |
|---|---|
| Event camera | Prophesee Gen 4.1 EVK4 (1280×720) |
| Frame camera | DJI Osmo Action 4 in webcam mode (30fps, 1280×720) |
| Kicker | Capacitor-discharge solenoid (tested at 40V, 80V, 120V DC) |
| Goalkeeper actuator | Linear actuator controlled via Arduino over USB serial |
| Sync | LED flash visible to both cameras for timestamp alignment |

---

## Dependencies

### EVK4 scripts
- [OpenEB](https://github.com/prophesee-ai/openeb) — Prophesee event camera SDK
- `metavision_core`, `metavision_sdk_core`, `metavision_sdk_ui`
- `opencv-python`
- `numpy`

### DJI scripts
- `opencv-python`
- `numpy`

Install DJI dependencies:
```bash
pip install opencv-python numpy
```

For EVK4, follow the [OpenEB installation guide](https://docs.prophesee.ai/stable/installation/index.html).

---

## Calibration

Run calibration before first use. Each camera requires its own `calibration.json`.

**EVK4:**
```bash
python tracking_only/evk4/calibrate_evk4.py
```
1. Drag ROI over playing field (exclude solenoid and LED)
2. Click left then right goalpost
3. Enter physical goal width in mm

**DJI:**
```bash
python tracking_only/dji/calibrate_dji.py
```
Same steps — uses live feed, press Space to snapshot before clicking.

**LED sync ROI** (run once per camera position):
```bash
python tracking_only/evk4/led_calibrate_evk4.py
python tracking_only/dji/led_calibrate_dji.py
```
Copy the printed `LED_ROI` value into the respective tracker script.

---

## Usage

### Tracking only (simultaneous dual-camera testing)

Run each script on a separate laptop connected to its respective camera.

**EVK4:**
```bash
python tracking_only/evk4/evk4_tracker.py
```

**DJI:**
```bash
python tracking_only/dji/dji_tracker.py
```

**Controls (both scripts):**

| Key | Action |
|---|---|
| `n` | Arm new shot — logging starts on LED flash |
| `s` | End shot — stops logging |
| `c` | Clear tracker history |
| `q` | Quit and save CSV log |

### Goalkeeper integration

```bash
python goalkeeper_integration/evk4/evk4_goalkeeper.py
python goalkeeper_integration/dji/dji_goalkeeper.py
```

Predicted intercept coordinates are sent to the Arduino over USB serial in mm from the left goalpost.

---

## Data Logging

Both tracking scripts write a CSV log per session:

| Column | Description |
|---|---|
| `event_ts_us` / `frame_number` | Hardware timestamp (EVK4) or frame count (DJI) |
| `wall_time_s` | System clock — used for inter-camera alignment via LED sync |
| `ball_x_px`, `ball_y_px` | Ball centroid in pixels |
| `ball_x_mm`, `ball_y_mm` | Ball position in mm from left goalpost |
| `intercept_mm` | Predicted goalkeeper target position (mm from left post) |
| `detection` | 1 if ball detected this window, 0 if not |
| `led_detected` | 1 on rising edge of LED flash (sync marker) |
| `shot_id` | Incremented each time `n` is pressed |

---

## Algorithm Overview

Both cameras implement an identical pipeline after their respective sensing and preprocessing stages:

1. **Ball detection** — contour detection on thresholded/masked frame, area-filtered to ball size
2. **CentroidTracker** — maintains deque of recent (x, y, timestamp) positions
3. **Intercept prediction** — linear polyfit over last 8 positions, extrapolated to goal line
4. **Output** — goalkeeper target coordinate in mm from left goalpost

Preprocessing differs by hardware necessity:
- **EVK4:** median blur + brightness threshold (event frame is inherently motion-selective)
- **DJI:** MOG2 background subtraction + HSV colour masking (compensates for static scene in RGB frame)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

> George, J. (2026). *Low-Latency Event Camera Demonstrator: Neuromorphic Robotic Foosball Goalkeeper System*. Queensland University of Technology. https://github.com/Jenna-G/foosball-event-camera-tracker
