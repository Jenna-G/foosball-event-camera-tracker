"""
Full DJI Foosball Calibration Tool
1. First: Select Table ROI (drag or 4 corners)
2. Then: Click Left and Right Goalposts
3. Enter real goal width in mm
"""

import cv2
import json
import numpy as np

OUTPUT_FILE = "calibration.json"

# Global variables
mode = 'drag'          # 'drag' or 'corners'
roi = None
corners = []
goalposts = []
step = 1               # 1=ROI, 2=Goalposts

def mouse_callback(event, x, y, flags, param):
    global roi, corners, goalposts, mode

    if step == 1:  # ROI Selection
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 'drag':
                corners = [(x, y)]  # Start new drag
            elif mode == 'corners' and len(corners) < 4:
                corners.append((x, y))
                print(f"Corner {len(corners)}/4: ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE and mode == 'drag' and len(corners) == 1:
            # Live preview while dragging
            pass

        elif event == cv2.EVENT_LBUTTONUP and mode == 'drag' and len(corners) == 1:
            x1, y1 = corners[0]
            if x > x1 and y > y1:
                roi = (x1, y1, x - x1, y - y1)
                print(f"ROI selected: {roi}")
            corners = []

    elif step == 2:  # Goalpost Selection
        if event == cv2.EVENT_LBUTTONDOWN and len(goalposts) < 2:
            goalposts.append((x, y))
            print(f"Goalpost {len(goalposts)}/2: ({x}, {y})")


def main():
    global step, mode, roi, corners, goalposts

    print("🔍 Opening DJI Camera for Calibration...\n")

    # Try to find DJI camera
    cap = None
    for idx in [1, 0, 2, 3]:
        for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"✅ Using camera index {idx}")
                break
        if cap.isOpened():
            break

    if not cap or not cap.isOpened():
        print("❌ Could not open camera")
        return

    # Take high quality snapshot
    for _ in range(5):  # Warm up camera
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image")
        return

    snapshot = frame.copy()
    print(f"Image size: {snapshot.shape[1]}x{snapshot.shape[0]}")

    cv2.namedWindow("Foosball Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Foosball Calibration", mouse_callback)

    print("\n=== STEP 1: Select Table ROI ===")
    print("Press 'd' → Drag mode | 'c' → Click 4 corners | 'r' → Reset | 'n' → Next step")

    while True:
        display = snapshot.copy()

        # Draw ROI
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Draw goalposts if in step 2
        if step == 2:
            for i, (x, y) in enumerate(goalposts):
                color = (0, 255, 255)
                cv2.circle(display, (x, y), 8, color, -1)
                label = "LEFT" if i == 0 else "RIGHT"
                cv2.putText(display, label, (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Foosball Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Cancelled.")
            break

        elif key == ord('d'):
            mode = 'drag'
            print("→ Drag mode")
        elif key == ord('c'):
            mode = 'corners'
            print("→ Corner mode (click 4 points)")
        elif key == ord('r'):
            roi = None
            corners = []
            goalposts = []
            print("→ Reset")
        elif key == ord('n') and step == 1:
            if roi is None:
                print("Please select ROI first!")
            else:
                step = 2
                print("\n=== STEP 2: Click Left then Right Goalpost ===")
                print("Press ENTER when done")

        elif key == 13 and step == 2 and len(goalposts) == 2:   # ENTER
            break

    cv2.destroyAllWindows()

    if roi is None or len(goalposts) != 2:
        print("Incomplete calibration.")
        return

    # ===================== SAVE CALIBRATION =====================
    goal_width_mm = float(input("\nEnter real distance between goalposts (mm): "))

    left = goalposts[0]
    right = goalposts[1]
    pixel_dist = np.hypot(right[0] - left[0], right[1] - left[1])
    px_per_mm = pixel_dist / goal_width_mm

    # Goal line equation
    x1, y1 = left
    x2, y2 = right
    if abs(x2 - x1) < 5:
        m = None
        c = x1
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

    calib = {
        "roi": roi,
        "left_post_px": list(left),
        "right_post_px": list(right),
        "goal_width_mm": round(goal_width_mm, 1),
        "px_per_mm": round(px_per_mm, 6),
        "goal_line_m": m,
        "goal_line_c": round(c, 4) if m is not None else c,
        "image_width": snapshot.shape[1],
        "image_height": snapshot.shape[0],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\n✅ Full calibration saved to '{OUTPUT_FILE}'")
    print(f"   ROI: {roi}")
    print(f"   Scale: {px_per_mm:.4f} px/mm")


if __name__ == "__main__":
    main()