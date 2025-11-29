# pi_app/run_edgeguard.py

import time
from pir_sensor import PIRSensor
from camera_inference import EdgeGuardInference
import cv2


def main():
    print("[INFO] Starting EdgeGuard AI Runtime...")

    # Initialize PIR sensor(turn sensor on and off)
    #pir = PIRSensor(pir_pin=4)

    # Initialize TFLite model
    ai = EdgeGuardInference(
        model_path="edgeguard_fp16.tflite",
        labels_path="labels.txt"
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the camera.")
        return

    print("[INFO] System ready. Waiting for motion...")

    while True:
        #pir.wait_for_motion()  # Blocking call until motion is detected(uncomment to run sensor)

        # Read frame when motion is detected
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to capture frame.")
            continue

        # Run inference
        label, conf = ai.predict(frame)
        print(f"[DETECT] {label} ({conf*100:.1f}%) at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Optional: display image with prediction
        annotated = ai.draw_prediction(frame, label, conf)
        cv2.imshow("EdgeGuard Live", annotated)

        # Show for a short period (500ms)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

        print("[INFO] Re-arming PIR sensor...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] EdgeGuard AI runtime stopped by user.")
    finally:
        cv2.destroyAllWindows()

