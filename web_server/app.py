#!/usr/bin/env python3
import os
import sys
import time
import threading
from flask import Flask, Response, jsonify, render_template

# ============================================
# FIX PYTHON PATH TO IMPORT FROM pi_app/
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PI_APP_DIR = os.path.join(BASE_DIR, "pi_app")
sys.path.append(PI_APP_DIR)

# --- Correct imports ---
from ssd_mobilenet_detector import SSDMobilenetDetector
from sort_tracker import SortTracker
from video_feed import VideoCamera

# --- Database functions ---
from db import init_db, log_event, get_latest_events


# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)


# ============================================
# Friendly uptime formatter
# ============================================
def format_uptime(seconds):
    seconds = int(seconds)

    intervals = (
        ('year',   31536000),
        ('month',  2592000),
        ('week',   604800),
        ('day',    86400),
        ('hour',   3600),
        ('min',    60),
        ('sec',    1)
    )

    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            result.append(f"{value} {name}{'s' if value != 1 else ''}")

        if len(result) == 2:  # Limit to 2 largest units
            break

    return " ".join(result) if result else "0 sec"


# ============================================
# CPU Temperature helper
# ============================================
def get_cpu_temp():
    try:
        out = os.popen("vcgencmd measure_temp").read()
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except:
        return "--"


# ============================================
# INITIALIZE DETECTOR + TRACKER + CAMERA
# ============================================
print("[WEB] Initializing SSD Mobilenet Detector...")
detector = SSDMobilenetDetector(
    model_path=os.path.join(PI_APP_DIR, "detect.tflite"),
    label_path=os.path.join(PI_APP_DIR, "labelmap.txt"),
    score_threshold=0.4
)

print("[WEB] Initializing SORT tracker...")
tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3)

print("[WEB] Starting camera...")
camera = VideoCamera(detector, tracker)


# Store generic status info
status_data = {
    "fps": 0.0,
    "confidence": 0.0,
    "person_count": 0,
    "last_seen": "--",
    "uptime": 0,
    "cpu_temp": "--",
}


# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """ Streams MJPEG video from VideoCamera.generate(). """
    return Response(
        camera.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """ Returns live JSON status for dashboard. """

    status_data["fps"] = camera.fps
    status_data["cpu_temp"] = get_cpu_temp()
    status_data["uptime"] += 1
    status_data["last_seen"] = time.strftime("%H:%M:%S")

    # Pull from VideoCamera tracking counters
    status_data["person_count"] = camera.person_count
    status_data["confidence"] = camera.person_conf

    # Friendly uptime
    formatted_uptime = format_uptime(status_data["uptime"])

    # Load history
    events = get_latest_events(20)
    event_strings = [
        f"[{ts}] {count} person(s) — conf={conf:.2f} — {fps:.1f} FPS"
        for (ts, count, conf, fps) in events
    ]

    return jsonify({
        **status_data,
        "uptime": formatted_uptime,
        "events": event_strings
    })


# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    print("[WEB] Initializing database...")
    init_db()

    print("[WEB] Flask server starting at http://192.168.12.249:5000/ or http://100.74.3.97:5000 ")
    app.run(host="100.74.3.97", port=5000, debug=False, threaded=True)
