import os
import sys
from flask import Flask, Response, jsonify, render_template
import cv2
import time
import threading

# ========================
# FIX IMPORT PATH
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PI_APP_DIR = os.path.join(BASE_DIR, "pi_app")
sys.path.append(PI_APP_DIR)

from ssd_detector import SSDPersonDetector
from run_ssd_edgeguard import Sort
from db import init_db, log_event, get_latest_events


app = Flask(__name__)

detector = None
tracker = None
output_frame = None
frame_lock = threading.Lock()

status_data = {
    "fps": 0.0,
    "confidence": 0.0,
    "person_count": 0,
    "last_seen": "--",
    "uptime": 0,
    "cpu_temp": "--",
}


def get_cpu_temp():
    try:
        out = (
            os.popen("vcgencmd measure_temp")
            .read()
            .replace("temp=", "")
            .replace("'C", "")
            .strip()
        )
        return float(out)
    except:
        return "--"


def camera_loop():
    global output_frame, status_data

    cap = cv2.VideoCapture(0)
    detector_instance = SSDPersonDetector()
    tracker_instance = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

    start_time = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frames += 1
        duration = time.time() - start_time
        if duration > 1.0:
            status_data["fps"] = frames / duration
            frames = 0
            start_time = time.time()

        detections = detector_instance.detect(frame)
        boxes_xyxy = []
        scores = []

        if detections:
            for det in detections:
                boxes_xyxy.append(det["box"])
                scores.append(det["score"])
        
        if boxes_xyxy:
            dets_for_tracker = []
            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box
                dets_for_tracker.append([x1, y1, x2, y2, score])

            tracks = tracker_instance.update(dets_for_tracker)
        else:
            tracks = tracker_instance.update([])

        person_count = 0
        best_conf = 0.0

        for tr in tracks:
            x1, y1, x2, y2, track_id = tr
            person_count += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {int(track_id)}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if scores:
                best_conf = max(scores)

        status_data["person_count"] = person_count
        status_data["confidence"] = float(best_conf) if scores else 0.0
        status_data["last_seen"] = time.strftime("%H:%M:%S")
        status_data["cpu_temp"] = get_cpu_temp()
        status_data["uptime"] += 1

        # Log event only when a person is detected
        if person_count > 0:
            log_event(person_count, best_conf, status_data["fps"])

        with frame_lock:
            output_frame = frame.copy()

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")  # Keep using your HTML file


@app.route("/video_feed")
def video_feed():
    def generate():
        global output_frame
        while True:
            with frame_lock:
                if output_frame is None:
                    continue

                ret, jpeg = cv2.imencode(".jpg", output_frame)
                if not ret:
                    continue

                frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    events = get_latest_events(20)
    event_strings = [
        f"[{ts}] {count} person(s) — conf {conf:.2f} — {fps:.1f} FPS"
        for (ts, count, conf, fps) in events
    ]

    return jsonify({**status_data, "events": event_strings})


if __name__ == "__main__":
    init_db()
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
