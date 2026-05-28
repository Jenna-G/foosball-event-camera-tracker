import csv
from datetime import datetime
import cv2
import json
import numpy as np
import time
import contextlib
from collections import deque
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, MTWindow, BaseWindow

# ============================================================================
# CONFIGURATION
# ============================================================================

SERIAL_NUMBER     = "00050960"
DELTA_T           = 10000   #10000 for testing 120V 2nd time    #33333  # Match DJI frame period (1/30 s = 33333 µs)
CALIBRATION_FILE  = "calibration.json"
SHOW_MASK_WINDOW  = False       # set to False to hide the mask debug window

MIN_BALL_AREA     = 143         # was 100, then was 50 for most testing but now ~10% of nominal 1426 px² (35mm ball)
MAX_BALL_AREA     = 2139        # was 3000 for most testing,150% of 1426 and helps reduce player rod noise
THRESHOLD         = 50
MAX_TRACK_HISTORY = 30

# Rod ROI mask — list of (x, y, w, h) rectangles to black out before detection.
# Measure these in pixels from the event camera frame.
# Add one entry per rod/player column you want to suppress.
ROD_ROIS = [
    # (x,   y,   w,   h),
    # (120,  0,  20, 480),  # example — uncomment and adjust per rod
]

# LED sync detection
# Set LED_ROI to the LED's pixel location in the event camera frame.
LED_ROI                  = (0, 318, 18, 262)   # <-- adjust to LED location
LED_MIN_AREA             = 50 #was 300 but reduced when doing DELTA_T=10000 testing


# ============================================================================
# LOGGING
# ============================================================================

def make_log_filename(prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.csv"


class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "event_ts_us", "wall_time_s",
            "ball_x_px", "ball_y_px", "ball_x_mm", "ball_y_mm",
            "intercept_mm", "detection", "led_detected",
            "shot_id", "notes"
        ])
        print(f"Logging to: {filename}")

    def write(self, event_ts_us, wall_time_s,
              ball_x_px, ball_y_px, ball_x_mm, ball_y_mm,
              intercept_mm, detection, led_detected,
              shot_id, notes=""):
        self.writer.writerow([
            event_ts_us,
            f"{wall_time_s:.6f}",
            ball_x_px if ball_x_px is not None else "",
            ball_y_px if ball_y_px is not None else "",
            f"{ball_x_mm:.2f}" if ball_x_mm is not None else "",
            f"{ball_y_mm:.2f}" if ball_y_mm is not None else "",
            f"{intercept_mm:.2f}" if intercept_mm is not None else "",
            detection, led_detected, shot_id, notes
        ])
        self.file.flush()

    def close(self):
        self.file.close()
        print(f"Log saved: {self.filename}")


# ============================================================================
# CALIBRATION
# ============================================================================

def load_calibration(path):
    with open(path) as f:
        return json.load(f)

# ============================================================================
# BALL DETECTION
# ============================================================================

def apply_rod_mask(gray):
    masked = gray.copy()
    for (x, y, w, h) in ROD_ROIS:
        masked[y:y+h, x:x+w] = 0
    return masked


def detect_ball_centroid(gray, event_ts_us, roi=None):
    # Uses hardware event timestamp, not system clock.
    # This is critical — tracker.positions will store true sensing times,
    # making velocity and latency calculations reflect actual hardware timing.
    event_ts_s = event_ts_us / 1_000_000.0
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    masked  = apply_rod_mask(gray)
    blurred = cv2.medianBlur(masked, 3)
    _, thresh = cv2.threshold(blurred, THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_blob = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BALL_AREA <= area <= MAX_BALL_AREA):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if area > best_area:
            best_blob = (int(cx), int(cy), int(radius))
            best_area = area

    if best_blob is None:
        return None, thresh

    x, y, radius = best_blob
    # Convert ROI coordinates back to full frame coordinates
    if roi is not None:
        rx, ry, rw, rh = roi
        x = x + rx
        y = y + ry
    return (x, y, radius, event_ts_s), thresh

LED_THRESHOLD = 40  # separate higher threshold just for LED detection

def detect_led(gray):
    x, y, w, h = LED_ROI
    roi = gray[y:y+h, x:x+w]
    _, led_thresh = cv2.threshold(roi, LED_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(led_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= LED_MIN_AREA:
            return True
    return False

def px_to_mm(x_px, y_px, calib):
    """Converts pixel position to mm offset from left post for logging."""
    lx, ly = calib["left_post_px"]
    x_mm = (x_px - lx) / calib["px_per_mm"]
    y_mm = (y_px - ly) / calib["px_per_mm"]
    return x_mm, y_mm

# ============================================================================
# TRACKING
# ============================================================================

class CentroidTracker:
    def __init__(self, max_history=MAX_TRACK_HISTORY):
        self.positions = deque(maxlen=max_history)
        self.last_detection_time = None

    def update(self, detection):
        if detection is not None:
            x, y, radius, timestamp = detection
            self.positions.append((x, y, timestamp))
            self.last_detection_time = timestamp
            return True
        return False

    def get_velocity(self):
        if len(self.positions) < 2:
            return None
        x1, y1, t1 = self.positions[-2]
        x2, y2, t2 = self.positions[-1]
        dt = t2 - t1
        if dt == 0:
            return None
        return (x2 - x1) / dt, (y2 - y1) / dt

    def is_tracking(self):
        if self.last_detection_time is None or len(self.positions) == 0:
            return False
        latest_ts = self.positions[-1][2]
        return (latest_ts - self.last_detection_time) < 0.5

# ============================================================================
# GOALKEEPER INTERCEPT
# ============================================================================

def compute_intercept(tracker, calib):
    """
    Fit a line through recent ball positions and find where it crosses
    the goalkeeper line. Returns (intercept_px, intercept_mm) or (None, None).
    intercept_mm is distance from the left post along the goal line.
    """
    if len(tracker.positions) < 3:
        return None, None

    pts = np.array([(x, y) for x, y, _ in tracker.positions], dtype=np.float32)
    pts = pts[-8:]  # use last 8 points for stable fit

    t = np.arange(len(pts), dtype=np.float32)
    px = np.polyfit(t, pts[:, 0], 1)  # x(t) = px[0]*t + px[1]
    py = np.polyfit(t, pts[:, 1], 1)  # y(t) = py[0]*t + py[1]

    vx, vy = px[0], py[0]
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return None, None  # ball stationary

    bx, by = pts[-1]  # current ball position

    m = calib["goal_line_m"]
    c = calib["goal_line_c"]

    if m is None:
        # Vertical goalkeeper line x = c
        goal_x = c
        if abs(vx) < 1e-3:
            return None, None
        s = (goal_x - bx) / vx
        goal_y = by + s * vy
    else:
        # Ball ray: x = bx + s*vx, y = by + s*vy
        # Goal line: y = m*x + c
        # Solve: by + s*vy = m*(bx + s*vx) + c
        denom = vy - m * vx
        if abs(denom) < 1e-3:
            return None, None  # trajectory parallel to goal line
        s = (m * bx + c - by) / denom
        goal_x = bx + s * vx
        goal_y = by + s * vy

    # Only report if intercept is in the future
    if s <= 0:
        return None, None

    # Only report if between the two posts
    lx, ly = calib["left_post_px"]
    rx, ry = calib["right_post_px"]
    min_x, max_x = min(lx, rx), max(lx, rx)
    if not (min_x <= goal_x <= max_x):
        return None, None

    intercept_px = (int(goal_x), int(goal_y))
    dist_px      = np.hypot(goal_x - lx, goal_y - ly)
    intercept_mm = dist_px / calib["px_per_mm"]

    return intercept_px, intercept_mm

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_tracking(frame, detection, tracker):
    if detection is not None:
        x, y, radius, _ = detection
        cv2.circle(frame, (x, y), max(radius, 6), (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"({x},{y})", (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if len(tracker.positions) > 1:
        points = [(int(x), int(y)) for x, y, _ in tracker.positions]
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)

    velocity = tracker.get_velocity()
    if velocity is not None and detection is not None:
        x, y, _, _ = detection
        vx, vy = velocity
        end_x = int(x + vx * 0.1)
        end_y = int(y + vy * 0.1)
        cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 255), 2)
        speed = np.hypot(vx, vy)
        cv2.putText(frame, f"v:{speed:.0f}px/s", (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def draw_calibration(frame, calib):
    lx, ly = calib["left_post_px"]
    rx, ry = calib["right_post_px"]
    cv2.line(frame, (lx, ly), (rx, ry), (0, 180, 255), 2)
    cv2.circle(frame, (lx, ly), 5, (0, 180, 255), -1)
    cv2.circle(frame, (rx, ry), 5, (0, 180, 255), -1)


def draw_intercept(frame, intercept_px):
    if intercept_px is not None:
        cv2.circle(frame, intercept_px, 10, (0, 0, 255), 3)
        cv2.circle(frame, intercept_px, 3,  (0, 0, 255), -1)
        cv2.putText(frame, "TARGET", (intercept_px[0] + 12, intercept_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_rod_rois(frame):
    for (x, y, w, h) in ROD_ROIS:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)

def draw_roi(frame, roi):
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_led_roi(frame, led_on):
    x, y, w, h = LED_ROI
    colour = (0, 255, 255) if led_on else (80, 80, 80)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    if led_on:
        cv2.putText(frame, "SYNC", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def build_mask_window(thresh, out_h, out_w):
    resized = cv2.resize(thresh, (out_w, out_h))
    bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, f"thresh={THRESHOLD}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return bgr

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    log_filename = make_log_filename("evk4_tracking")
    logger = DataLogger(log_filename)

    try:
        calib = load_calibration(CALIBRATION_FILE)
        if "roi" in calib and calib["roi"] is not None:
            ROI = tuple(calib["roi"])
            print(f"ROI loaded: {ROI}")
        else:
            print("No ROI in calibration — using full frame")
            ROI = None
        print(f"Calibration loaded: {calib['px_per_mm']:.4f} px/mm, "
              f"goal width {calib['goal_width_mm']:.0f} mm")
    except FileNotFoundError:
        print(f"WARNING: {CALIBRATION_FILE} not found — run calibrate.py first.")
        print("Continuing without goalkeeper prediction.")
        calib = None
        ROI = None
        
    print("\nControls:")
    print("  n  — arm new shot (increments shot_id, clears track — logging starts on LED flash)")
    print("  s  — end shot (stops logging)")
    print("  c  — clear track only")
    print("  q  — quit and save log\n")

    mv_iterator = EventsIterator(input_path=SERIAL_NUMBER, delta_t=DELTA_T)
    height, width = mv_iterator.get_size()

    frame_gen = OnDemandFrameGenerationAlgorithm(width, height, palette=ColorPalette.Dark)
    tmp_2d    = np.zeros((height, width, 3), dtype=np.uint8)

    tracker      = CentroidTracker()
    keep_running = True
    frame_count  = 0
    start_time   = time.time()
    last_intercept_mm = None
    state = {"shot_id": 0, "shot_active": False, "led_was_on": False, "armed": False}

    def keyboard_cb(key, scancode, action, mods):
        nonlocal keep_running, tracker
        if action == 0:   # ignore key release — callback fires on both press and release
            return
        if key in [ord("q"), ord("Q"), 27]:
            keep_running = False
        elif key in [ord("c"), ord("C")]:
            tracker = CentroidTracker()
            print("Track cleared")
        elif key in [ord("n"), ord("N")]:
            state["shot_id"]    += 1
            state["shot_active"] = False
            state["armed"]       = True
            tracker              = CentroidTracker()
            print(f"--- Shot {state['shot_id']} armed — flash LED to begin logging ---")
        elif key in [ord("s"), ord("S")]:
            state["shot_active"] = False
            state["armed"]       = False
            state["led_was_on"]  = False
            print(f"--- Shot {state['shot_id']} ended ---")

    mask_h = height // 2
    mask_w = width

    mask_ctx = MTWindow(title="Mask Debug",
                        width=mask_w, height=mask_h,
                        mode=BaseWindow.RenderMode.BGR) if SHOW_MASK_WINDOW else contextlib.nullcontext()

    with MTWindow(title="Foosball Event Tracker",
                  width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as main_window, \
         mask_ctx as mask_window:

        main_window.set_keyboard_callback(keyboard_cb)
        if SHOW_MASK_WINDOW and mask_window is not None:
            mask_window.set_keyboard_callback(keyboard_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            if not keep_running or main_window.should_close() or \
               (mask_window is not None and mask_window.should_close()):
                break
            if evs.size == 0:
                continue

            frame_count += 1

            wall_time   = time.time()
            event_ts_us = int(evs["t"][-1])

            frame_gen.process_events(evs)
            tmp_2d.fill(0)
            frame_gen.generate(event_ts_us, tmp_2d)

            gray = cv2.cvtColor(tmp_2d, cv2.COLOR_BGR2GRAY)
            detection, thresh = detect_ball_centroid(gray, event_ts_us, ROI)
            tracker.update(detection)

            # LED sync rising edge detection:
            led_on    = detect_led(gray)    #uses full frame gray, not cropped thresh
            led_event = 1 if (led_on and not state["led_was_on"]) else 0
            if led_event and state["armed"]:
                state["shot_active"] = True
                state["armed"]       = False
                print(f"[Shot {state['shot_id']}] LED SYNC — logging started at "
                      f"{event_ts_us} us (wall: {wall_time:.6f})")
            state["led_was_on"] = led_on

            # Goalkeeper intercept — only compute when ball is actively detected
            intercept_px, intercept_mm = None, None
            if calib is not None and detection is not None:
                intercept_px, intercept_mm = compute_intercept(tracker, calib)

            if intercept_mm != last_intercept_mm:
                if intercept_mm is not None and state["shot_active"]:
                    print(f"[Shot {state['shot_id']}] GOALKEEPER TARGET: {intercept_mm:.1f} mm")
                last_intercept_mm = intercept_mm

            # ball position in mm:
            ball_x_px = ball_y_px = ball_x_mm = ball_y_mm = None
            if detection is not None:
                ball_x_px, ball_y_px = detection[0], detection[1]
                if calib is not None:
                    ball_x_mm, ball_y_mm = px_to_mm(ball_x_px, ball_y_px, calib)

            # write log row:
            if state["shot_active"]:
                logger.write(
                    event_ts_us=event_ts_us,
                    wall_time_s=wall_time,
                    ball_x_px=ball_x_px,
                    ball_y_px=ball_y_px,
                    ball_x_mm=ball_x_mm,
                    ball_y_mm=ball_y_mm,
                    intercept_mm=intercept_mm,
                    detection=1 if detection is not None else 0,
                    led_detected=led_event,
                    shot_id=state["shot_id"]
                )

            if calib is not None:
                draw_calibration(tmp_2d, calib)
            draw_rod_rois(tmp_2d)
            draw_roi(tmp_2d, ROI)
            draw_tracking(tmp_2d, detection, tracker)
            draw_intercept(tmp_2d, intercept_px)
            draw_led_roi(tmp_2d, led_on)

            fps    = frame_count / (time.time() - start_time)
            status = "TRACKING" if tracker.is_tracking() else "LOST"
            cv2.putText(tmp_2d, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(tmp_2d, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if tracker.is_tracking() else (0, 0, 255), 2)
            cv2.putText(tmp_2d, f"Track length: {len(tracker.positions)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if state["shot_active"]:
                shot_label  = f"Shot:{state['shot_id']} [REC]"
                shot_colour = (0, 0, 255)
            elif state["armed"]:
                shot_label  = f"Shot:{state['shot_id']} [ARMED]"
                shot_colour = (0, 165, 255)
            else:
                shot_label  = f"Shot:{state['shot_id']} [IDLE]"
                shot_colour = (200, 200, 0)
            cv2.putText(tmp_2d, shot_label, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, shot_colour, 2)

            if intercept_mm is not None:
                cv2.putText(tmp_2d, f"Target: {intercept_mm:.1f} mm", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            main_window.show_async(tmp_2d)
            if mask_window is not None:
                mask_window.show_async(build_mask_window(thresh, mask_h, mask_w))

    logger.close()
    elapsed = time.time() - start_time
    print(f"\nSession complete: {frame_count} frames, {frame_count/elapsed:.1f} FPS avg")


if __name__ == "__main__":
    main()