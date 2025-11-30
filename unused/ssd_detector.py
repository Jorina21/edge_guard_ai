# ssd_detector.py
# Clean detector wrapper for SSD model used by Flask server

import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tflite


class SSDPersonDetector:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "person_ssd.tflite")
        label_path = os.path.join(os.path.dirname(__file__), "labelmap.txt")

        # Load labels
        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def detect(self, frame):
        """Runs SSD detection & returns person boxes in xyxy format"""

        # Preprocess
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_tensor = np.expand_dims(resized, axis=0)

        # Set tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Extract outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []

        # Iterate through results
        for i in range(len(scores)):
            if scores[i] < 0.5:
                continue

            class_id = int(classes[i])
            if class_id != 1:  # person class = 1
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            h, w, _ = frame.shape
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "class_id": class_id
            })

        return detections
