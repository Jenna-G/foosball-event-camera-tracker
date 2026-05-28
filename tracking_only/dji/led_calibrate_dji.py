"""
led_calibrate_dji.py — Run this once to find the LED ROI for the DJI camera.

Instructions:
1. Position your sync LED where it will sit during testing.
2. Turn the LED ON so it is visible.
3. The live DJI feed will appear — press SPACE to grab a snapshot.
4. Click the TOP-LEFT corner of the LED, then the BOTTOM-RIGHT corner.
5. The script prints the LED_ROI value — copy it into DJI_centroid_tracking_v7.py.

Does NOT modify calibration.json.
"""

import cv2
import numpy as np

CAMERA_INDEX   = 1
CAMERA_BACKEND = cv2.CAP_DSHOW

clicks  = []
snapshot = None

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"  Click {len(clicks)}: ({x}, {y})")

def main():
    global snapshot

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print("ERROR: Cannot open DJI camera.")
        return

    print("Make sure the sync LED is ON before continuing.")
    print("Live feed is showing — press SPACE to grab a snapshot, Q to quit.\n")

    cv2.namedWindow("LED Calibration — DJI")

    # Live preview until SPACE pressed
    while snapshot is None:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        display = frame.copy()
        cv2.putText(display, "Press SPACE to snapshot, Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("LED Calibration — DJI", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            snapshot = frame.copy()
            print("Snapshot captured.\n")
        elif key in [ord("q"), 27]:
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()

    print("Click the TOP-LEFT corner of the LED, then the BOTTOM-RIGHT corner.")
    print("Press R to reset clicks, ENTER to confirm, Q to quit.\n")

    cv2.setMouseCallback("LED Calibration — DJI", on_click)

    while True:
        display = snapshot.copy()

        for i, (x, y) in enumerate(clicks):
            cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
            label = ["TOP-LEFT", "BOTTOM-RIGHT"][i]
            cv2.putText(display, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if len(clicks) == 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            roi_x = min(x1, x2)
            roi_y = min(y1, y2)
            roi_w = abs(x2 - x1)
            roi_h = abs(y2 - y1)
            cv2.rectangle(display, (roi_x, roi_y),
                          (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.putText(display,
                        f"LED_ROI = ({roi_x}, {roi_y}, {roi_w}, {roi_h})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, "Press ENTER to confirm, R to reset",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

        cv2.imshow("LED Calibration — DJI", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(clicks) == 2:   # ENTER
            break
        elif key in [ord("r"), ord("R")]:    # R — reset
            clicks.clear()
            print("Clicks reset.")
        elif key in [ord("q"), 27]:          # Q / ESC
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    x1, y1 = clicks[0]
    x2, y2 = clicks[1]
    roi_x = min(x1, x2)
    roi_y = min(y1, y2)
    roi_w = abs(x2 - x1)
    roi_h = abs(y2 - y1)

    print("\n" + "="*55)
    print("Copy this value into DJI_centroid_tracking_v7.py:")
    print(f"\n    LED_ROI = ({roi_x}, {roi_y}, {roi_w}, {roi_h})\n")
    print("="*55)

if __name__ == "__main__":
    main()
