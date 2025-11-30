# yolo_detector.py
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time
from datetime import datetime


class YOLOv8Detector:
    def __init__(self,
                 model_path="yolov8n_float32.tflite",
                 input_size=640,
                 conf_threshold=0.4,
                 iou_threshold=0.45,
                 detection_callback=None):

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detection_callback = detection_callback
        self.last_fps = 0.0

        print("[YOLOv8] Loading TFLite model:", model_path)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("[YOLOv8] Model ready.")

    # -------------------------
    # Preprocess frame
    # -------------------------
    def preprocess(self, frame):
        h, w, _ = frame.shape
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img, h, w

    # -------------------------
    # NMS
    # -------------------------
    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return keep

    # -------------------------
    # YOLO Inference
    # -------------------------
    def detect(self, frame):

        start_total = time.time()
        img, orig_h, orig_w = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        start_infer = time.time()
        self.interpreter.invoke()
        infer_ms = (time.time() - start_infer) * 1000

        raw = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        raw = raw.transpose(1, 0)  # [8400, 84]

        boxes = []
        scores = []
        class_ids = []

        for det in raw:
            xc, yc, w, h = det[:4]
            obj_conf = det[4]

            if obj_conf < self.conf_threshold:
                continue

            class_probs = det[5:]
            cid = np.argmax(class_probs)
            cls_conf = class_probs[cid]

            conf = obj_conf * cls_conf
            if conf < self.conf_threshold:
                continue

            # Only person class
            if cid != 0:
                continue

            # Convert xywh -> xyxy
            x1 = int((xc - w / 2) * orig_w / self.input_size)
            y1 = int((yc - h / 2) * orig_h / self.input_size)
            x2 = int((xc + w / 2) * orig_w / self.input_size)
            y2 = int((yc + h / 2) * orig_h / self.input_size)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
            class_ids.append(cid)

        keep = self.nms(boxes, scores)

        results = []
        for i in keep:
            results.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i]
            })

        # FPS Calculation
        total_time = time.time() - start_total
        if total_time > 0:
            self.last_fps = 1.0 / total_time

        # Update callback for Flask
        if self.detection_callback is not None:
            if len(results) > 0:
                best = max(results, key=lambda x: x["score"])
                self.detection_callback({
                    "person": True,
                    "confidence": best["score"],
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            else:
                self.detection_callback({
                    "person": False,
                    "confidence": 0.0,
                    "timestamp": None
                })

        return results, infer_ms

    # -------------------------
    # Draw boxes
    # -------------------------
    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Person {score*100:.1f}%",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        return frame

