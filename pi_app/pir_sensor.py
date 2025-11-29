# pi_app/pir_sensor.py

from gpiozero import MotionSensor
import time

class PIRSensor:
    def __init__(self, pir_pin=4):   # default GPIO pin 4
        print("[INFO] Initializing PIR sensor on GPIO", pir_pin)
        self.sensor = MotionSensor(pir_pin)

    def wait_for_motion(self):
        """
        Blocks until PIR detects motion.
        Returns the timestamp of detection.
        """
        print("[INFO] Waiting for motion...")
        self.sensor.wait_for_motion()
        t = time.time()
        print(f"[INFO] Motion detected at {t}")
        return t

    def motion_detected(self):
        """
        Checks if motion is currently detected.
        Returns True/False.
        """
        return self.sensor.motion_detected

