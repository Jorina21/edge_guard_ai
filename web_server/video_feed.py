# video_feed.py
import cv2
import time
import numpy as np

class VideoCamera:
    def __init__(self, detector, tracker):
        """
        detector = SSDMobilenetDetector instance
        tracker  = SortTracker instance
        """
        self.detector = detector
        self.tracker = tracker

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.0

    def _update_fps(self):
        self.frames += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frames / (now - self.last_time)
            self.frames = 0
            self.last_time = now

    def generate(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self._update_fps()

            # ------------------------------------------------------
            # STEP 1: SSD PERSON DETECTION
            # ------------------------------------------------------
            detections = self.detector.detect(frame)
            
            # Convert to numpy array for SORT
            if detections:
                det_array = np.array([
                    [*d["box"], d["score"]]   # x1, y1, x2, y2, score
                    for d in detections
                ], dtype=float)
            else:
                det_array = np.empty((0, 5))

            # ------------------------------------------------------
            # STEP 2: SORT TRACKING
            # ------------------------------------------------------
            tracks = self.tracker.update(det_array)

            # ------------------------------------------------------
            # STEP 3: DRAW BOXES + IDs
            # ------------------------------------------------------
            for trk in tracks:
                x1, y1, x2, y2 = trk["box"]
                tid = trk["id"]
                score = trk["score"]

                # Draw bounding box
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)

                cv2.putText(
                    frame,
                    f"ID {tid} ({score*100:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Draw FPS
            cv2.putText(frame,
                        f"FPS: {self.fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

            # ------------------------------------------------------
            # STEP 4: ENCODE MJPEG FRAME
            # ------------------------------------------------------
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )
