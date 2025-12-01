# video_feed.py
import cv2
import time
import numpy as np

from db import log_event


class VideoCamera:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # FPS counters
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.0

        # For dashboard
        self.person_count = 0
        self.person_conf = 0.0

        # For logging cooldown
        self.last_event_time = 0

        # Last frame for streaming
        self.last_frame = None

    # -------------------------------
    # FPS COUNTER
    # -------------------------------
    def _update_fps(self):
        self.frames += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frames / (now - self.last_time)
            self.frames = 0
            self.last_time = now

    # -------------------------------
    # BACKGROUND LOOP FOR DASHBOARD
    # -------------------------------
    def background_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self._update_fps()

            # 1. DETECT
            detections = self.detector.detect(frame)

            if detections:
                det_array = np.array([[*d["box"], d["score"]] for d in detections], dtype=float)
            else:
                det_array = np.empty((0, 5))

            # 2. TRACK
            tracks = self.tracker.update(det_array)

            # 3. SAVE METRICS
            self.person_count = len(tracks)
            self.person_conf = max([t["score"] for t in tracks], default=0.0)

            # 4. LOG EVENTS (2s cooldown)
            now = time.time()
            if self.person_count > 0 and (now - self.last_event_time) > 2:
                log_event(self.person_count, self.person_conf, self.fps)
                self.last_event_time = now

            # 5. SAVE FRAME FOR STREAMING
            ret, jpeg = cv2.imencode(".jpg", frame)
            if ret:
                self.last_frame = jpeg.tobytes()

    # -------------------------------
    # MJPEG STREAMING LOOP
    # -------------------------------
    def generate(self):
        while True:
            if self.last_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    self.last_frame +
                    b"\r\n"
                )
            time.sleep(0.03)
