import cv2
import json
import numpy as np
import time
import contextlib
from collections import deque
import serial
import serial.tools.list_ports
import threading
import queue

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, MTWindow, BaseWindow

# ===================== CONFIG =====================
TEST_MODE = False                  # ← Set to False for full automatic tracking

SHOW_THRESHOLD = False             # ← Set to True if you want the threshold debug window
SHOW_MAIN_WINDOW = False            # ← Main tracking window (recommended: True)

SERIAL_PORT   = "COM6"
BAUD_RATE     = 115200
SEND_INTERVAL = 0.045

SERIAL_NUMBER     = "00050960"
DELTA_T           = 33333
CALIBRATION_FILE  = "calibration.json"

MIN_BALL_AREA     = 143
MAX_BALL_AREA     = 2139
THRESHOLD         = 50
MAX_TRACK_HISTORY = 30

ROD_ROIS = [
    # (x, y, w, h) — add your foosball rod regions here
]

LED_ROI                  = (0, 318, 18, 262)
LED_MIN_AREA             = 50
LED_THRESHOLD            = 40

# ========================== MANUAL COMMAND QUEUE ==========================
command_queue = queue.Queue()

def input_thread():
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
# CALIBRATION + DETECTION + TRACKING (unchanged from your working version)
# ============================================================================

def load_calibration(path):
    with open(path) as f:
        return json.load(f)

def apply_rod_mask(gray):
    masked = gray.copy()
    for (x, y, w, h) in ROD_ROIS:
        masked[y:y+h, x:x+w] = 0
    return masked

def detect_ball_centroid(gray, event_ts_us, roi=None):
    event_ts_s = event_ts_us / 1_000_000.0
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    masked = apply_rod_mask(gray)
    blurred = cv2.medianBlur(masked, 3)
    _, thresh = cv2.threshold(blurred, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_blob = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_BALL_AREA <= area <= MAX_BALL_AREA:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if area > best_area:
                best_blob = (int(cx), int(cy), int(radius))
                best_area = area
    if best_blob is None:
        return None, thresh
    x, y, radius = best_blob
    if roi is not None:
        rx, ry, rw, rh = roi
        x = x + rx
        y = y + ry
    return (x, y, radius, event_ts_s), thresh

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
    lx, ly = calib["left_post_px"]
    x_mm = (x_px - lx) / calib["px_per_mm"]
    y_mm = (y_px - ly) / calib["px_per_mm"]
    return x_mm, y_mm


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


def compute_intercept(tracker, calib):
    if len(tracker.positions) < 3:
        return None, None
    pts = np.array([(x, y) for x, y, _ in tracker.positions], dtype=np.float32)
    pts = pts[-8:]
    t = np.arange(len(pts), dtype=np.float32)
    px = np.polyfit(t, pts[:, 0], 1)
    py = np.polyfit(t, pts[:, 1], 1)
    vx, vy = px[0], py[0]
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return None, None
    bx, by = pts[-1]
    m = calib.get("goal_line_m")
    c = calib.get("goal_line_c")
    if m is None:
        goal_x = c
        if abs(vx) < 1e-3:
            return None, None
        s = (goal_x - bx) / vx
        goal_y = by + s * vy
    else:
        denom = vy - m * vx
        if abs(denom) < 1e-3:
            return None, None
        s = (m * bx + c - by) / denom
        goal_x = bx + s * vx
        goal_y = by + s * vy
    if s <= 0:
        return None, None
    lx, ly = calib["left_post_px"]
    rx, ry = calib["right_post_px"]
    min_x, max_x = min(lx, rx), max(lx, rx)
    if not (min_x <= goal_x <= max_x):
        return None, None
    intercept_px = (int(goal_x), int(goal_y))
    dist_px = np.hypot(goal_x - lx, goal_y - ly)
    intercept_mm = dist_px / calib["px_per_mm"]
    return intercept_px, intercept_mm


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
        cv2.circle(frame, intercept_px, 3, (0, 0, 255), -1)
        cv2.putText(frame, "TARGET", (intercept_px[0] + 12, intercept_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_rod_rois(frame):
    for (x, y, w, h) in ROD_ROIS:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)


def draw_led_roi(frame, led_on):
    x, y, w, h = LED_ROI
    colour = (0, 255, 255) if led_on else (80, 80, 80)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5, write_timeout=0.5)
        print(f"✅ Successfully connected to Arduino on {SERIAL_PORT}")
        time.sleep(2.0)
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return

    threading.Thread(target=input_thread, daemon=False).start()

    if TEST_MODE:
        print("🔧 TEST MODE ACTIVE - Only testing Arduino connection")
    else:
        print("🚀 Starting full event camera + automatic goalie tracking...")

    try:
        if TEST_MODE:
            while True:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd == "QUIT":
                        raise KeyboardInterrupt
                    ser.write(f"{cmd}\n".encode())
                    print(f"→ Sent: {cmd}")
                time.sleep(0.05)
        else:
            calib = load_calibration(CALIBRATION_FILE)
            tracker = CentroidTracker()

            mv_iterator = EventsIterator(input_path=SERIAL_NUMBER, delta_t=DELTA_T)
            height, width = mv_iterator.get_size()
            frame_gen = OnDemandFrameGenerationAlgorithm(width, height, palette=ColorPalette.Dark)

            last_send_time = time.time()
            frame_skip = 0

            print("✅ Camera stream started successfully!")

            for evs in mv_iterator:
                frame_gen.process_events(evs)
                tmp_2d = np.zeros((height, width, 3), dtype=np.uint8)
                frame_gen.generate(int(evs["t"][-1]), tmp_2d)

                gray = cv2.cvtColor(tmp_2d, cv2.COLOR_BGR2GRAY)
                detection, thresh = detect_ball_centroid(gray, int(evs["t"][-1]))

                tracker.update(detection)
                intercept_px, intercept_mm = compute_intercept(tracker, calib)

                # Automatic goalie control
                if intercept_mm is not None and (time.time() - last_send_time > SEND_INTERVAL):
                    try:
                        cmd = f"{int(intercept_mm)}\n"
                        ser.write(cmd.encode())
                        print(f"→ AUTO GOALIE: {intercept_mm:.1f} mm")
                        last_send_time = time.time()
                    except:
                        pass

                # Draw visuals (less often to reduce lag)
                frame_skip = (frame_skip + 1) % 3
                if frame_skip == 0:
                    draw_tracking(tmp_2d, detection, tracker)
                    draw_calibration(tmp_2d, calib)
                    draw_intercept(tmp_2d, intercept_px)
                    draw_rod_rois(tmp_2d)
                    draw_led_roi(tmp_2d, detect_led(gray))

                    if SHOW_MAIN_WINDOW:
                        cv2.imshow("EVK4 Ball Tracker", tmp_2d)
                    if SHOW_THRESHOLD:
                        cv2.imshow("Threshold", thresh)

                # Manual override
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd == "QUIT":
                        raise KeyboardInterrupt
                    try:
                        ser.write(f"{cmd}\n".encode())
                        print(f"→ Manual: {cmd}")
                    except:
                        pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()
        if ser and ser.is_open:
            ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()