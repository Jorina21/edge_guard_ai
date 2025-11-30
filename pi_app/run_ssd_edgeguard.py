#!/usr/bin/env python3
import cv2
import time
from ssd_mobilenet_detector import SSDMobilenetDetector


def main():
    print("[RUNTIME] Starting SSD Mobilenet EdgeGuard AI...")

    # --- Load SSD Mobilenet detector ---
    detector = SSDMobilenetDetector(
        model_path="detect.tflite",
        label_path="labelmap.txt",
        score_threshold=0.4
    )

    # --- Initialize Camera ---
    print("[RUNTIME] Initializing camera...")
    cap = cv2.VideoCapture(0)

    # Performance settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[RUNTIME] Camera ready. Press 'q' to quit.")

    # FPS tracking
    last_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        frame_count += 1
        now = time.time()
        elapsed = now - last_time

        # --- Every ~0.5 sec, update FPS display ---
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            last_time = now
        else:
            fps = None

        # --- Run detection ---
        detections = detector.detect(frame)

        # --- Draw boxes ---
        annotated = detector.draw(frame, detections)

        # --- Put FPS on screen ---
        if fps is not None:
            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # --- Show window ---
        cv2.imshow("EdgeGuard AI - SSD Mobilenet", annotated)

        # --- Exit on q ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[RUNTIME] Exiting EdgeGuard AI...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
