# video_feed.py
import cv2
from flask import Response

class VideoCamera:
    def __init__(self, detector):
        self.detector = detector
        self.cap = cv2.VideoCapture(0)

        # Force correct resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def generate(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Run YOLO detection
            detections, infer_ms = self.detector.detect(frame)
            frame = self.detector.draw(frame, detections)

            # Encode for MJPEG streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

