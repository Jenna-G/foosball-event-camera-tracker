import csv
from datetime import datetime
import cv2
import json
import numpy as np
import time
from collections import deque
import serial
import serial.tools.list_ports
import threading
import queue

# ============================================================================
# CONFIGURATION - Calibrate these for your setup
# ============================================================================

CAMERA_INDEX = 0  #either 0 or 1
CAMERA_BACKEND = cv2.CAP_DSHOW
CALIBRATION_FILE = "calibration.json"
SHOW_MASK_WINDOW  = False       # set to False to hide the mask debug window

# Ball detection parameters (based on your tuned values)
BALL_HSV_LOWER = np.array([80, 35, 125])    # Adjusted lower bound
BALL_HSV_UPPER = np.array([105, 90, 220])   # Adjusted upper bound

MIN_BALL_AREA = 52   # ~30% of nominal 172 px² (px_per_mm=0.4250, 35mm ball)
MAX_BALL_AREA = 500
MAX_TRACK_HISTORY = 30

# ROI will be loaded from calibration.json
ROI = None

# LED sync detection
LED_ROI      = (501, 566, 281, 62)   # <-- adjust using led_calibrate_dji.py
LED_MIN_AREA = 200

# ========================== SERIAL CONFIG ==========================
SERIAL_PORT   = "COM6"
BAUD_RATE     = 115200
SEND_INTERVAL = 0.045          # ~22 Hz — safe for Arduino

# ========================== MANUAL COMMAND QUEUE ==========================
command_queue = queue.Queue()

def input_thread():
    """Background thread for manual terminal commands while vision loop runs."""
    print("\n" + "="*60)
    print("MANUAL CONTROL TERMINAL READY")
    print("="*60)
    print("Commands: 150, stop, center, home, 0, q")
    print("="*60 + "\n")
    while True:
        try:
            cmd = input(">> ").strip()
            if not cmd:
                continue
            if cmd.lower() == 'q':
                command_queue.put("QUIT")
                break
            command_queue.put(cmd)
        except:
            break


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
            "frame_number", "wall_time_s",
            "ball_x_px", "ball_y_px", "ball_x_mm", "ball_y_mm",
            "intercept_mm", "detection", "led_detected",
            "shot_id", "notes"
        ])
        print(f"Logging to: {filename}")

    def write(self, frame_number, wall_time_s,
              ball_x_px, ball_y_px, ball_x_mm, ball_y_mm,
              intercept_mm, detection, led_detected,
              shot_id, notes=""):
        self.writer.writerow([
            frame_number,
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
# BALL DETECTION - Find ball centroid using motion and color within ROI
# ============================================================================

def detect_ball_centroid(frame, bg_subtractor, roi):
    timestamp = time.time()
    
    # Apply ROI if defined
    if roi is not None:
        x, y, w, h = roi
        if x + w <= frame.shape[1] and y + h <= frame.shape[0]:
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame
    else:
        roi_frame = frame
    
    # Update background model
    fg_mask = bg_subtractor.apply(roi_frame)
    
    # Apply threshold to refine motion mask
    fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate to connect broken parts of the ball
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    
    # Convert to HSV and apply color mask
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    ball_mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    
    # Combine motion mask with color mask
    mask = cv2.bitwise_and(ball_mask, ball_mask, mask=fg_mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, mask
    
    best_blob = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_BALL_AREA <= area <= MAX_BALL_AREA:
            (x_roi, y_roi), radius = cv2.minEnclosingCircle(contour)
            if area > best_area:
                best_blob = (int(x_roi), int(y_roi), int(radius))
                best_area = area
    
    if best_blob is None:
        return None, mask
    
    x_roi, y_roi, radius = best_blob
    # Convert ROI coordinates back to full frame coordinates
    x = x_roi + (roi[0] if roi else 0)
    y = y_roi + (roi[1] if roi else 0)
    return (x, y, radius, timestamp), mask


def detect_led(frame):
    """Safe LED detection with empty frame protection"""
    if frame is None or frame.size == 0:
        return False
    x, y, w, h = LED_ROI
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return False
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return False
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, led_thresh = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY)
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
# TRACKING - Maintain ball position history
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
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        return (vx, vy)
    
    def get_track_length(self):
        return len(self.positions)
    
    def is_tracking(self):
        if self.last_detection_time is None:
            return False
        time_since_detection = time.time() - self.last_detection_time
        return time_since_detection < 0.5


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
    px = np.polyfit(t, pts[:, 0], 1)
    py = np.polyfit(t, pts[:, 1], 1)

    vx, vy = px[0], py[0]
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return None, None

    bx, by = pts[-1]

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
        # Goal line: y = m*x + c  →  solve for s
        denom = vy - m * vx
        if abs(denom) < 1e-3:
            return None, None
        s = (m * bx + c - by) / denom
        goal_x = bx + s * vx
        goal_y = by + s * vy

    if s <= 0:
        return None, None  # intercept is behind the ball

    lx, ly = calib["left_post_px"]
    rx, ry = calib["right_post_px"]
    min_x, max_x = min(lx, rx), max(lx, rx)
    if not (min_x <= goal_x <= max_x):
        return None, None  # outside the goal

    intercept_px = (int(goal_x), int(goal_y))
    dist_px      = np.hypot(goal_x - lx, goal_y - ly)
    intercept_mm = dist_px / calib["px_per_mm"]

    return intercept_px, intercept_mm


# ============================================================================
# VISUALIZATION - Draw tracking results
# ============================================================================

def draw_tracking(frame, detection, tracker, roi):
    # Draw ROI
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if detection is not None:
        x, y, radius, _ = detection
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"({x},{y})", (x + 15, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if tracker.get_track_length() > 1:
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
        speed = np.sqrt(vx**2 + vy**2)
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


def draw_led_roi(frame, led_on):
    x, y, w, h = LED_ROI
    colour = (0, 255, 255) if led_on else (80, 80, 80)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    if led_on:
        cv2.putText(frame, "SYNC", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Serial setup
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5, write_timeout=0.5)
        print(f"✅ Successfully connected to Arduino on {SERIAL_PORT}")
        time.sleep(2.0)
        threading.Thread(target=input_thread, daemon=False).start()
    except Exception as e:
        print(f"⚠️ Serial failed: {e} (continuing without Arduino)")

    # Load calibration
    log_filename = make_log_filename("dji_goalie")
    logger = DataLogger(log_filename)

    try:
        calib = load_calibration(CALIBRATION_FILE)
        print(f"✅ Calibration loaded: {calib['px_per_mm']:.4f} px/mm, "
              f"goal width {calib.get('goal_width_mm', '?')} mm")
        
        if "roi" in calib and calib["roi"] is not None:
            ROI = tuple(calib["roi"])
            print(f"✅ ROI loaded from calibration: {ROI}")
        else:
            print("⚠️  No ROI found in calibration.json — using full frame")
            ROI = None
    except FileNotFoundError:
        print(f"WARNING: {CALIBRATION_FILE} not found")
        calib = None
        ROI = None

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Camera opened successfully. Press 'b' to capture background snapshot (remove ball first).")
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=False)
    
    # Capture background snapshot
    background_frame = None
    while background_frame is None:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to read frame")
            continue
        cv2.imshow('Setup', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            background_frame = frame.copy()
            print("Background snapshot captured. Starting tracking...")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    if background_frame is not None:
        for _ in range(30):
            bg_subtractor.apply(background_frame)
    
    tracker = CentroidTracker()
    frame_count = 0
    start_time = time.time()
    last_send_time = 0
    state = {"shot_id": 0, "shot_active": False, "led_was_on": False, "armed": False}

    print("\nControls:")
    print("  b  — capture background (remove ball first)")
    print("  n  — arm new shot (logging starts on LED flash)")
    print("  s  — end shot")
    print("  c  — clear track only")
    print("  q  — quit and save log\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        frame_count += 1
        wall_time = time.time()
        
        # Process manual commands
        while not command_queue.empty():
            cmd = command_queue.get_nowait()
            if cmd == "QUIT":
                cap.release()
                cv2.destroyAllWindows()
                if ser and ser.is_open:
                    ser.close()
                return
            if ser and ser.is_open:
                ser.write(f"{cmd}\n".encode())
                print(f"→ Sent: {cmd}")

        # Detect ball
        detection, mask = detect_ball_centroid(frame, bg_subtractor, ROI)

        # LED sync (safe call)
        led_on = detect_led(frame)
        led_event = 1 if (led_on and not state["led_was_on"]) else 0
        if led_event and state["armed"]:
            state["shot_active"] = True
            state["armed"] = False
            print(f"[Shot {state['shot_id']}] LED SYNC — logging started")
        state["led_was_on"] = led_on

        tracker.update(detection)

        # Automatic goalie control
        intercept_px, intercept_mm = None, None
        if calib is not None:
            intercept_px, intercept_mm = compute_intercept(tracker, calib)
            if intercept_mm is not None and (time.time() - last_send_time > SEND_INTERVAL):
                goal_width = calib.get("goal_width_mm", 160.0)
                position = int(12 + (intercept_mm / goal_width) * (160 - 12))
                if ser and ser.is_open:
                    ser.write(f"{position}\n".encode())
                    print(f"→ AUTO GOALIE: {position} mm")
                last_send_time = time.time()

        # Logging (if active)
        if state["shot_active"] and logger:
            ball_x_px = ball_y_px = ball_x_mm = ball_y_mm = None
            if detection is not None:
                ball_x_px, ball_y_px = detection[0], detection[1]
                if calib is not None:
                    ball_x_mm, ball_y_mm = px_to_mm(ball_x_px, ball_y_px, calib)
            logger.write(frame_count, wall_time, ball_x_px, ball_y_px,
                         ball_x_mm, ball_y_mm, intercept_mm,
                         1 if detection else 0, led_event, state["shot_id"])

        # Visualization
        draw_tracking(frame, detection, tracker, ROI)
        if calib:
            draw_calibration(frame, calib)
        draw_intercept(frame, intercept_px)
        draw_led_roi(frame, led_on)

        cv2.imshow('DJI Goalie Tracker', frame)
        if SHOW_MASK_WINDOW:
            cv2.imshow('Detection Mask', mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            print("Background snapshot captured")
        elif key == ord('n'):
            state["shot_id"] += 1
            state["shot_active"] = False
            state["armed"] = True
            tracker = CentroidTracker()
            print(f"--- Shot {state['shot_id']} armed — flash LED to begin logging ---")
        elif key == ord('s'):
            state["shot_active"] = False
            state["armed"] = False
            state["led_was_on"] = False
            print(f"--- Shot {state['shot_id']} ended ---")
        elif key == ord('c'):
            tracker = CentroidTracker()
            print("Track cleared")

    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
    if logger:
        logger.close()
    print("Program ended.")


if __name__ == "__main__":
    main()