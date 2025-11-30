# run_yolo_edgeguard.py
# Run YOLOv8 TFLite person detection + simple tracking on Raspberry Pi

import cv2
import time

from yolo8_detector import YOLOv8Detector
from simple_tracker import SimpleTracker


def main():
    print("[RUNTIME] Starting YOLOv8 EdgeGuard AI...")

    # --- Initialize YOLOv8 detector ---
    detector = YOLOv8Detector(
        model_path="yolov8n_fp16.tflite",  # or "yolov8n_float16.tflite" if you prefer
        input_size=640,
        conf_threshold=0.4,
        iou_threshold=0.45,
    )

    # --- Initialize simple IoU-based tracker ---
    tracker = SimpleTracker(iou_threshold=0.3, max_lost=10)

    # --- Open camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return
    
    # Force lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    print("[RUNTIME] Press 'q' to quit.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            continue

        # --- Run YOLOv8 detection ---
        detections, infer_ms = detector.detect(frame)  # list of {box, score, class_id}

        # --- Update tracker with detections ---
        # Tracker expects list of dicts with key 'box'
        tracks = tracker.update(detections)  # list of {id, box}

        # --- Draw detection boxes (green) ---
        frame = detector.draw(frame, detections)

        # --- Draw tracking IDs on top of boxes (yellow) ---
        for tr in tracks:
            tid = tr["id"]
            x1, y1, x2, y2 = tr["box"]
            cv2.putText(
                frame,
                f"ID {tid}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # --- FPS estimate ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"{len(tracks)} person(s), {infer_ms:.1f} ms, {fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # --- Show window ---
        cv2.imshow("EdgeGuard AI - YOLOv8 Detection & Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[RUNTIME] Stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[RUNTIME] Interrupted by user.")
        cv2.destroyAllWindows()

