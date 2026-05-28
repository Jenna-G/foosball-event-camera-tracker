"""
calibrate_evk4_full.py — Run this once before the tracker.

Instructions:
1. A snapshot from the event camera will appear.
2. Drag to select the table ROI (excludes LED and other areas outside the field).
3. Click the LEFT goalpost, then the RIGHT goalpost.
4. Enter the real-world distance between them in mm when prompted.
5. Calibration is saved to calibration.json.
"""

import cv2
import json
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette

SERIAL_NUMBER  = "00050960"
OUTPUT_FILE    = "calibration.json"
SNAPSHOT_DELTA = 100000  # grab a single 100ms frame for clicking on

# ============================================================================
# SHARED STATE
# ============================================================================

state = {
    "step":       1,       # 1 = ROI selection, 2 = goalpost selection
    "roi":        None,    # (x, y, w, h)
    "drag_start": None,    # first corner of drag
    "drag_end":   None,    # current drag end point (for live preview)
    "goalposts":  [],      # list of (x, y) clicks, max 2
}

# ============================================================================
# MOUSE CALLBACK
# ============================================================================

def mouse_callback(event, x, y, flags, param):
    s = state

    if s["step"] == 1:
        if event == cv2.EVENT_LBUTTONDOWN:
            s["drag_start"] = (x, y)
            s["drag_end"]   = None

        elif event == cv2.EVENT_MOUSEMOVE and s["drag_start"] is not None:
            s["drag_end"] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and s["drag_start"] is not None:
            x1, y1 = s["drag_start"]
            if x > x1 and y > y1:
                s["roi"] = (x1, y1, x - x1, y - y1)
                print(f"  ROI selected: {s['roi']}")
            s["drag_start"] = None
            s["drag_end"]   = None

    elif s["step"] == 2:
        if event == cv2.EVENT_LBUTTONDOWN and len(s["goalposts"]) < 2:
            s["goalposts"].append((x, y))
            label = ["LEFT post", "RIGHT post"][len(s["goalposts"]) - 1]
            print(f"  Click {len(s['goalposts'])}: {label} ({x}, {y})")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Grabbing snapshot from event camera...")
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
    print(f"Snapshot captured: {width}x{height}\n")

    cv2.namedWindow("EVK4 Calibration")
    cv2.setMouseCallback("EVK4 Calibration", mouse_callback)

    # -------------------------------------------------------------------------
    # STEP 1 — ROI selection
    # -------------------------------------------------------------------------
    print("=== STEP 1: Select Table ROI ===")
    print("Drag to draw a rectangle over the playing field.")
    print("Exclude the LED and anything outside the table surface.")
    print("Press R to reset, N to confirm and move to goalposts, Q to quit.\n")

    while True:
        display = snapshot.copy()

        # Draw confirmed ROI
        if state["roi"] is not None:
            x, y, w, h = state["roi"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"ROI: {state['roi']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw live drag preview
        if state["drag_start"] is not None and state["drag_end"] is not None:
            cv2.rectangle(display, state["drag_start"], state["drag_end"],
                          (0, 200, 255), 1)

        cv2.putText(display, "STEP 1: Drag ROI | R=reset | N=next | Q=quit",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("EVK4 Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        elif key in [ord("r"), ord("R")]:
            state["roi"]        = None
            state["drag_start"] = None
            state["drag_end"]   = None
            print("ROI reset.")
        elif key in [ord("n"), ord("N")]:
            if state["roi"] is None:
                print("Please select a ROI first.")
            else:
                state["step"] = 2
                break

    # -------------------------------------------------------------------------
    # STEP 2 — Goalpost selection
    # -------------------------------------------------------------------------
    print("\n=== STEP 2: Click Left then Right Goalpost ===")
    print("Press R to reset clicks, ENTER to confirm, Q to quit.\n")

    while True:
        display = snapshot.copy()

        # Draw ROI for reference
        x, y, w, h = state["roi"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Draw goalpost clicks
        for i, (gx, gy) in enumerate(state["goalposts"]):
            cv2.circle(display, (gx, gy), 6, (0, 255, 255), -1)
            label = ["LEFT post", "RIGHT post"][i]
            cv2.putText(display, label, (gx + 8, gy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(state["goalposts"]) == 2:
            cv2.line(display, state["goalposts"][0], state["goalposts"][1],
                     (0, 255, 0), 2)
            cv2.putText(display, "Goalkeeper line — press ENTER to confirm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, "STEP 2: Click goalposts | R=reset | ENTER=confirm | Q=quit",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("EVK4 Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        elif key in [ord("r"), ord("R")]:
            state["goalposts"] = []
            print("Goalpost clicks reset.")
        elif key == 13 and len(state["goalposts"]) == 2:   # ENTER
            break

    cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # COMPUTE AND SAVE
    # -------------------------------------------------------------------------
    goal_width_mm = float(input("\nEnter the real-world distance between the goalposts in mm: "))

    left_px, right_px = state["goalposts"][0], state["goalposts"][1]
    pixel_distance = np.hypot(right_px[0] - left_px[0], right_px[1] - left_px[1])
    px_per_mm = pixel_distance / goal_width_mm

    x1, y1 = left_px
    x2, y2 = right_px
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
    else:
        m = None
        c = x1

    calib = {
        "roi":           list(state["roi"]),
        "left_post_px":  list(left_px),
        "right_post_px": list(right_px),
        "goal_width_mm": goal_width_mm,
        "px_per_mm":     px_per_mm,
        "goal_line_m":   m,
        "goal_line_c":   c,
        "image_width":   width,
        "image_height":  height,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\nCalibration saved to {OUTPUT_FILE}")
    print(f"  ROI:             {state['roi']}")
    print(f"  Pixel distance:  {pixel_distance:.1f} px")
    print(f"  Scale:           {px_per_mm:.4f} px/mm")
    if m is not None:
        print(f"  Goalkeeper line: y = {m:.4f}x + {c:.1f}")
    else:
        print(f"  Goalkeeper line: vertical x = {c}")

if __name__ == "__main__":
    main()
