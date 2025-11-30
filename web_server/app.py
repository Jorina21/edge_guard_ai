#!/usr/bin/env python3
import os
import sys
import time
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

# Global status for dashboard
status_data = {
    "fps": 0.0,
    "confidence": 0.0,
    "person_count": 0,
    "last_seen": "--",
    "uptime": 0,
    "cpu_temp": "--",
}


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



# ============================================
# ROUTES
# ============================================
@app.route("/")
def index():
    return render_template("index.html")



@app.route("/video_feed")
def video_feed():
    """
    Route streaming MJPEG frames from VideoCamera.generate()
    """
    return Response(
        camera.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/status")
def status():
    """
    Returns dashboard JSON containing:
    - FPS
    - Person count
    - Latest events from SQLite
    - CPU temp
    - Uptime
    """
    # Update live server metrics
    status_data["fps"] = camera.fps
    status_data["cpu_temp"] = get_cpu_temp()
    status_data["uptime"] += 1

    # Person count tracking via SORT
    status_data["person_count"] = camera.person_count if hasattr(camera.tracker, "last_count") else 0
    status_data["confidence"] = camera.person_conf if hasattr(camera.tracker, "last_conf") else 0.0
    status_data["last_seen"] = time.strftime("%H:%M:%S")

    # Grab last 20 events from DB
    events = get_latest_events(20)
    event_strings = [
        f"[{ts}] {count} person(s) — conf={conf:.2f} — {fps:.1f} FPS"
        for (ts, count, conf, fps) in events
    ]

    return jsonify({**status_data, "events": event_strings})



# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    print("[WEB] Initializing database...")
    init_db()

    print("[WEB] Flask server starting at http://0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
