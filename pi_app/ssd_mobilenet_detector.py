# ssd_mobilenet_detector.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class SSDMobilenetDetector:
    def __init__(self,
                 model_path="detect.tflite",
                 label_path="labelmap.txt",
                 score_threshold=0.4):

        self.score_threshold = score_threshold

        # Load labels
        with open(label_path, "r") as f:
            self.labels = [l.strip() for l in f.readlines()]

        # Load interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # SSD expects 300x300 input
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        print("[SSD] Ready. Model loaded:", model_path)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_expanded = np.expand_dims(img_rgb, axis=0)
        return img_expanded

    def detect(self, frame):
        """Returns list of {box:[], score, class_id}"""
        input_tensor = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        h, w, _ = frame.shape

        results = []
        for i, score in enumerate(scores):
            if score < self.score_threshold:
                continue

            class_id = int(classes[i])
            label = self.labels[class_id]

            # Only detect PERSON
            if label != "person":
                continue

            y1, x1, y2, x2 = boxes[i]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)

            results.append({
                "box": [x1, y1, x2, y2],
                "score": float(score),
                "class_id": class_id
            })

        return results

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame,
                        f"Person {score*100:.1f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)
        return frame
