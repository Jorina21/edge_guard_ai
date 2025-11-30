import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

MODEL = "detect.tflite"
LABELS = "labelmap.txt"

# Load labels
with open(LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("[INFO] Labels loaded:", len(labels))

# Load model
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("[INFO] Model loaded. Input:", input_details[0]['shape'])

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("[INFO] Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    # Preprocess
    img = cv2.resize(frame, (300, 300))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # Extract outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Print TOP 5 results
    print("\n--- RAW OUTPUT ---")
    for i in range(5):
        print(f"#{i}: score={scores[i]:.3f} class={classes[i]} label={labels[int(classes[i])] if int(classes[i]) < len(labels) else '???'}")

    # Draw detections
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < 0.3:
            continue

        class_id = int(classes[i])

        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        label = labels[class_id] if class_id < len(labels) else f"ID {class_id}"
        score = scores[i]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0), 2)

    cv2.imshow("SSD DEBUG VIEW", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
