#!/usr/bin/env python3
import os
import sys
import time
import threading
from flask import Flask, Response, jsonify, render_template

# Path fixes
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PI_APP_DIR = os.path.join(BASE_DIR, "pi_app")
sys.path.append(PI_APP_DIR)

# Imports
from ssd_mobilenet_detector import SSDMobilenetDetector
from sort_tracker import SortTracker
from video_feed import VideoCamera
from db import init_db, log_event, get_latest_events


app = Flask(__name__)

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
        out = os.popen("vcgencmd measure_temp").read()
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except:
        return "--"


print("[WEB] Loading SSD Mobilenet...")
detector = SSDMobilenetDetector(
    model_path=os.path.join(PI_APP_DIR, "detect.tflite"),
    label_path=os.path.join(PI_APP_DIR, "labelmap.txt"),
    score_threshold=0.4
)

print("[WEB] Loading SORT tracker...")
tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3)

print("[WEB] Starting camera module...")
camera = VideoCamera(detector, tracker)


# ===============================
# Routes
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        camera.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    status_data["fps"] = camera.fps
    status_data["cpu_temp"] = get_cpu_temp()
    status_data["uptime"] += 1

    status_data["person_count"] = camera.person_count
    status_data["confidence"] = camera.person_conf
    status_data["last_seen"] = time.strftime("%H:%M:%S")

    events = get_latest_events(20)
    event_strings = [
        f"[{ts}] {count} person(s) — conf={conf:.2f} — {fps:.1f} FPS"
        for (ts, count, conf, fps) in events
    ]

    return jsonify({**status_data, "events": event_strings})


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("[WEB] Initializing DB...")
    init_db()

    print("[WEB] Starting camera background thread...")
    threading.Thread(target=camera.background_loop, daemon=True).start()

    print("[WEB] Running Flask at 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
