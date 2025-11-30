# pi_app/simple_tracker.py

import numpy as np

class SimpleTracker:
    """
    Very lightweight tracker:
      - Keeps track of active tracks with IDs
      - Matches new detections to old ones using IoU
      - Creates new IDs when no match is found
    """

    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.next_id = 1
        self.tracks = {}  # id -> { 'box': [x1,y1,x2,y2], 'lost': int }
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH

        if inter == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / float(areaA + areaB - inter + 1e-6)

    def update(self, detections):
        """
        detections: list of dicts with key 'box' -> [x1,y1,x2,y2]
        Returns list of dicts: { 'id': int, 'box': [...], 'score': float }
        """
        updated_tracks = {}
        det_boxes = [d["box"] for d in detections]

        # Track assignment: greedy IoU matching
        for track_id, track_info in self.tracks.items():
            best_iou = 0.0
            best_idx = -1
            for i, box in enumerate(det_boxes):
                iou = self._iou(track_info["box"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > self.iou_threshold and best_idx >= 0:
                # Match found
                updated_tracks[track_id] = {
                    "box": det_boxes[best_idx],
                    "lost": 0,
                }
                det_boxes[best_idx] = None  # mark as used
            else:
                # No match – increase lost counter
                track_info["lost"] += 1
                if track_info["lost"] <= self.max_lost:
                    updated_tracks[track_id] = track_info

        # Create new tracks for unmatched detections
        for box in det_boxes:
            if box is None:
                continue
            track_id = self.next_id
            self.next_id += 1
            updated_tracks[track_id] = {"box": box, "lost": 0}

        self.tracks = updated_tracks

        # Return a list of track data
        outputs = []
        for tid, info in self.tracks.items():
            outputs.append({"id": tid, "box": info["box"]})

        return outputs

