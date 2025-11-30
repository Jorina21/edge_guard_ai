# yolo8_detector.py
# YOLOv8 TFLite Detector for EdgeGuard AI

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time


class YOLOv8Detector:
    def __init__(
        self,
        model_path="yolov8n_fp16.tflite",
        input_size=640,
        conf_threshold=0.25,   # lowered from 0.4
        iou_threshold=0.45
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print("[YOLOv8] Loading TFLite model:", model_path)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("[YOLOv8] Model loaded. Ready.")

    def preprocess(self, frame):
        # Resize to model input, normalize to [0,1]
        h, w, _ = frame.shape
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img, h, w

    def nms(self, boxes, scores):
        """Non-Maximum Suppression"""
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
        while len(order) > 0:
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

    def detect(self, frame):
        img, orig_h, orig_w = self.preprocess(frame)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        start = time.time()
        self.interpreter.invoke()
        infer_time = (time.time() - start) * 1000.0

        # Output is [1, 84, 8400] for YOLOv8n
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # [84, 8400]
        output = output.transpose(1, 0)  # → [8400, 84]

        boxes = []
        scores = []
        class_ids = []

        for det in output:
            # det[0:4] = [x_center, y_center, w, h] in *normalized* [0,1] coords
            x_center, y_center, w, h = det[0], det[1], det[2], det[3]

            # YOLOv8 logits at det[4:] → need sigmoid
            obj_logit = det[4]
            obj_conf = 1.0 / (1.0 + np.exp(-obj_logit))  # sigmoid

            if obj_conf < self.conf_threshold:
                continue

            class_logits = det[5:]
            class_probs = 1.0 / (1.0 + np.exp(-class_logits))  # sigmoid
            class_id = int(np.argmax(class_probs))
            class_conf = class_probs[class_id]

            final_conf = obj_conf * class_conf
            if final_conf < self.conf_threshold:
                continue

            # Only PERSON (COCO class 0)
            if class_id != 0:
                continue

            # Convert normalized xywh → pixel xyxy (no / self.input_size!)
            x1 = int((x_center - w / 2.0) * orig_w)
            y1 = int((y_center - h / 2.0) * orig_h)
            x2 = int((x_center + w / 2.0) * orig_w)
            y2 = int((y_center + h / 2.0) * orig_h)

            # Clamp to image bounds
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(float(final_conf))
            class_ids.append(class_id)

        # Apply NMS
        keep = self.nms(boxes, scores)

        results = []
        for i in keep:
            results.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i]
            })

        return results, infer_time

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {score*100:.1f}%",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        return frame

