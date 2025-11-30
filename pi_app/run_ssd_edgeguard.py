#!/usr/bin/env python3
import cv2
import time
import numpy as np

from ssd_mobilenet_detector import SSDMobilenetDetector
from sort_tracker import SortTracker


# --------------------------------------
# Deterministic colors for each track ID
# --------------------------------------
def id_to_color(track_id):
    # Ordered list of colors (BGR format for OpenCV)
    palette = [
        (0, 255, 0),      # 1 → Green
        (0, 255, 255),    # 2 → Yellow
        (0, 0, 255),      # 3 → Red
        (255, 0, 0),      # 4 → Blue
        (255, 255, 0),    # 5 → Cyan
        (255, 0, 255),    # 6 → Magenta
        (0, 128, 255),    # 7 → Orange
        (203, 192, 255),  # 8 → Pink
        (128, 0, 128),    # 9 → Purple
        (255, 255, 153),  # 10 → Light Blue-ish
    ]

    # Loop palette if track_id > 10
    index = (track_id - 1) % len(palette)
    return palette[index]

# --------------------------------------
# Utility: remove overlapping person boxes
# --------------------------------------
def suppress_overlaps(detections, iou_thresh=0.4):
    """
    Input: list of {box, score, class_id}
    Output: filtered list
    """

    if len(detections) <= 1:
        return detections

    boxes = np.array([det["box"] for det in detections])
    scores = np.array([det["score"] for det in detections])

    # Sort by highest confidence first
    order = scores.argsort()[::-1]

    keep = []
    suppressed = set()

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        if area_a + area_b - inter == 0:
            return 0
        return inter / (area_a + area_b - inter)

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)

        for j in order:
            if j == i or j in suppressed:
                continue
            if iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)

    return [detections[i] for i in keep]


# --------------------------------------
# MAIN RUNTIME
# --------------------------------------
def main():
    print("[RUNTIME] Starting SSD + SORT EdgeGuard AI...")

    # Detector
    detector = SSDMobilenetDetector(
        model_path="detect.tflite",
        label_path="labelmap.txt",
        score_threshold=0.3
    )

    # SORT Tracker
    tracker = SortTracker(
        max_age=10,
        min_hits=3,
        iou_threshold=0.3
    )

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[RUNTIME] Press 'q' to quit.")

    last_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        frame_count += 1
        now = time.time()
        if now - last_time >= 0.5:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        # Step 1: Detect people
        detections = detector.detect(frame)

        # Step 2: Remove duplicate person boxes
        detections = suppress_overlaps(detections)

        # Prepare SORT detections
        if len(detections) > 0:
            det_array = np.array([
                [
                    det["box"][0],
                    det["box"][1],
                    det["box"][2],
                    det["box"][3],
                    det["score"]
                ]
                for det in detections
            ], dtype=float)
        else:
            det_array = np.empty((0, 5), dtype=float)

        # Step 3: Update SORT tracker
        tracks = tracker.update(det_array)

        # Step 4: Draw tracked persons
        for trk in tracks:
            x1, y1, x2, y2 = trk["box"]
            track_id = trk["id"]
            score = trk["score"]

            color = id_to_color(track_id)

            cv2.rectangle(frame,
                          (x1, y1), (x2, y2),
                          color,
                          3)

            label = f"ID {track_id} ({score*100:.1f}%)"
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color, 2)

        # Step 5: HUD text
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"Persons: {len(tracks)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # Show
        cv2.imshow("EdgeGuard AI - SSD + SORT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
