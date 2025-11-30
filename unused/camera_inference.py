# pi_app/camera_inference.py

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class EdgeGuardInference:
    def __init__(self,
                 model_path="edgeguard_fp16.tflite",
                 labels_path="labels.txt",
                 input_size=(224, 224)):

        self.model_path = model_path
        self.labels = self.load_labels(labels_path)
        self.input_size = input_size

        # Load TFLite model
        print("[INFO] Loading TFLite model...")
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        print("[INFO] Model loaded.")

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

        print("[INFO] Input index:", self.input_index)
        print("[INFO] Output index:", self.output_index)

    def load_labels(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        input_tensor = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)[0]
        class_id = np.argmax(output)
        confidence = float(output[class_id])

        return self.labels[class_id], confidence

    def draw_prediction(self, frame, label, conf):
        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        return frame


def run_live_inference():
    model = EdgeGuardInference(
        model_path="edgeguard_fp16.tflite",
        labels_path="labels.txt"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    print("[INFO] Starting live inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        label, conf = model.predict(frame)
        frame = model.draw_prediction(frame, label, conf)

        cv2.imshow("EdgeGuard AI - Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_inference()

