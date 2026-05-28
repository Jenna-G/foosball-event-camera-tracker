"""
led_calibrate_evk4.py — Run this once to find the LED ROI for the event camera.

Instructions:
1. Position your sync LED where it will sit during testing.
2. Turn the LED ON so it is visible.
3. A snapshot from the event camera will appear.
4. Click the TOP-LEFT corner of the LED, then the BOTTOM-RIGHT corner.
5. The script prints the LED_ROI value — copy it into EVK4_ball_trackerV16.py.

Does NOT modify calibration.json.
"""

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette

SERIAL_NUMBER  = "00050960"
SNAPSHOT_DELTA = 100000   # grab a single 100ms accumulated frame

clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"  Click {len(clicks)}: ({x}, {y})")

def main():
    print("Make sure the sync LED is ON before continuing.")
    print("Grabbing snapshot from event camera...\n")

    mv_iterator = EventsIterator(input_path=SERIAL_NUMBER, delta_t=SNAPSHOT_DELTA)
    height, width = mv_iterator.get_size()
    frame_gen = OnDemandFrameGenerationAlgorithm(width, height, palette=ColorPalette.Dark)
    tmp_2d = np.zeros((height, width, 3), dtype=np.uint8)

    for evs in mv_iterator:
        if evs.size > 0:
            frame_gen.process_events(evs)
            frame_gen.generate(evs["t"][-1], tmp_2d)
        break  # one batch only

    snapshot = tmp_2d.copy()

    print("Click the TOP-LEFT corner of the LED, then the BOTTOM-RIGHT corner.")
    print("Press R to reset clicks, ENTER to confirm, Q to quit.\n")

    cv2.namedWindow("LED Calibration — EVK4")
    cv2.setMouseCallback("LED Calibration — EVK4", on_click)

    while True:
        display = snapshot.copy()

        # Draw clicks so far
        for i, (x, y) in enumerate(clicks):
            cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
            label = ["TOP-LEFT", "BOTTOM-RIGHT"][i]
            cv2.putText(display, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw ROI rectangle once both corners clicked
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

        cv2.imshow("LED Calibration — EVK4", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(clicks) == 2:   # ENTER — confirm
            break
        elif key in [ord("r"), ord("R")]:    # R — reset
            clicks.clear()
            print("Clicks reset.")
        elif key in [ord("q"), 27]:          # Q / ESC — quit
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
    print("Copy this value into EVK4_ball_trackerV16.py:")
    print(f"\n    LED_ROI = ({roi_x}, {roi_y}, {roi_w}, {roi_h})\n")
    print("="*55)

if __name__ == "__main__":
    main()
