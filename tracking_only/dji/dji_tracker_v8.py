import csv
from datetime import datetime
import cv2
import json
import numpy as np
import time
from collections import deque

# ============================================================================
# CONFIGURATION - Calibrate these for your setup
# ============================================================================

CAMERA_INDEX = 1  #either 0 or 1
CAMERA_BACKEND = cv2.CAP_DSHOW
CALIBRATION_FILE = "calibration.json"
SHOW_MASK_WINDOW  = False       # set to False to hide the mask debug window

# Ball detection parameters (based on your tuned values)
BALL_HSV_LOWER = np.array([75, 25, 100])    # Adjusted lower bound
BALL_HSV_UPPER = np.array([110, 110, 240])   # Adjusted upper bound

MIN_BALL_AREA = 13   # ~10% of nominal 129 px² for 35mm ball diameter
MAX_BALL_AREA = 194  # ~150%
MAX_TRACK_HISTORY = 30

# ROI will be loaded from calibration.json
ROI = None

# LED sync detection
LED_ROI      = (529, 552, 76, 10)   #(519, 543, 96, 17)   # <-- adjust using led_calibrate_dji.py
LED_MIN_AREA = 15

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
        if x + w <= frame.shape[1] and y + h <= frame.shape[0]:  # Ensure ROI is within frame
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame  # Use full frame if ROI is invalid
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
    x, y, w, h = LED_ROI
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, led_thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
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
    
    def get_current_position(self):
        if len(self.positions) > 0:
            return self.positions[-1]
        return None
    
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
# MAIN LOOP
# ============================================================================

def main():
    # Load calibration
    log_filename = make_log_filename("dji_tracking")
    logger = DataLogger(log_filename)
    try:
        calib = load_calibration(CALIBRATION_FILE)
        print(f"✅ Calibration loaded: {calib['px_per_mm']:.4f} px/mm, "
              f"goal width {calib['goal_width_mm']:.0f} mm")
        
        # Load ROI from calibration file
        if "roi" in calib and calib["roi"] is not None:
            ROI = tuple(calib["roi"])
            print(f"✅ ROI loaded from calibration: {ROI}")
        else:
            print("⚠️  No ROI found in calibration.json — using full frame")
            ROI = None
            
    except FileNotFoundError:
        print(f"WARNING: {CALIBRATION_FILE} not found — run calibrate_dji.py first.")
        print("Continuing without goalkeeper prediction.")
        calib = None
        ROI = None

    print("\nControls:")
    print("  b  — capture background (do this first)")
    print("  n  — arm new shot (logging starts on LED flash)")
    print("  s  — end shot (stops logging)")
    print("  c  — clear track only")
    print("  q  — quit and save log\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("Camera opened successfully. Press 'b' to capture background snapshot (remove ball first).")
    
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=False)
    
    # Capture background snapshot
    background_frame = None
    while background_frame is None:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
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
    
    # Warm up the background model with the snapshot
    if background_frame is not None:
        for _ in range(30):
            bg_subtractor.apply(background_frame)
    cv2.destroyWindow('Setup')

    tracker = CentroidTracker()
    frame_count = 0
    start_time = time.time()
    last_intercept_mm = None
    state = {"shot_id": 0, "shot_active": False, "led_was_on": False, "armed": False}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        wall_time = time.time()
        
        # Detect ball centroid with motion within ROI
        detection, mask = detect_ball_centroid(frame, bg_subtractor, ROI)
        tracker.update(detection)
            
        # Shot testing state logic
        led_on    = detect_led(frame)
        led_event = 1 if (led_on and not state["led_was_on"]) else 0
        if led_event and state["armed"]:
            state["shot_active"] = True
            state["armed"]       = False
            print(f"[Shot {state['shot_id']}] LED SYNC — logging started at "
                  f"frame {frame_count} (wall: {wall_time:.6f})")
        state["led_was_on"] = led_on

        # Goalkeeper intercept
        intercept_px, intercept_mm = None, None
        if calib is not None and detection is not None:
            intercept_px, intercept_mm = compute_intercept(tracker, calib)

        if intercept_mm != last_intercept_mm:
            if intercept_mm is not None and state["shot_active"]:
                print(f"[Shot {state['shot_id']}] GOALKEEPER TARGET: {intercept_mm:.1f} mm")
            last_intercept_mm = intercept_mm
        
        # mm conversion and data logging
        ball_x_px = ball_y_px = ball_x_mm = ball_y_mm = None
        if detection is not None:
            ball_x_px, ball_y_px = detection[0], detection[1]
            if calib is not None:
                ball_x_mm, ball_y_mm = px_to_mm(ball_x_px, ball_y_px, calib)

        if state["shot_active"]:
            logger.write(
                frame_number=frame_count,
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

        # Draw visualization
        if calib is not None:
            draw_calibration(frame, calib)
        draw_tracking(frame, detection, tracker, ROI)
        draw_intercept(frame, intercept_px)
        draw_led_roi(frame, led_on)
        
        fps = frame_count / (time.time() - start_time)
        status = "TRACKING" if tracker.is_tracking() else "LOST"
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if tracker.is_tracking() else (0, 0, 255), 2)
        cv2.putText(frame, f"Track length: {tracker.get_track_length()}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Shot state overlay
        if state["shot_active"]:
            shot_label  = f"Shot:{state['shot_id']} [REC]"
            shot_colour = (0, 0, 255)
        elif state["armed"]:
            shot_label  = f"Shot:{state['shot_id']} [ARMED]"
            shot_colour = (0, 165, 255)
        else:
            shot_label  = f"Shot:{state['shot_id']} [IDLE]"
            shot_colour = (200, 200, 0)
        cv2.putText(frame, shot_label, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, shot_colour, 2)

        if intercept_mm is not None:
            cv2.putText(frame, f"Target: {intercept_mm:.1f} mm", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Centroid Tracking', frame)
        if SHOW_MASK_WINDOW:
            cv2.imshow('Detection Mask', mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            filename = f"dji_shot{state['shot_id']}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('c'):
            tracker = CentroidTracker()
            print("Track cleared")
        elif key == ord('n'):
            state["shot_id"]    += 1
            state["shot_active"] = False
            state["armed"]       = True
            tracker              = CentroidTracker()
            print(f"--- Shot {state['shot_id']} armed — flash LED to begin logging ---")
        elif key == ord('s'):
            state["shot_active"] = False
            state["armed"]       = False
            state["led_was_on"]  = False
            print(f"--- Shot {state['shot_id']} ended ---")
    
    logger.close()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"\nSession complete:")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    main()
