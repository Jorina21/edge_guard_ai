# sort_tracker.py
# Lightweight SORT-style multi-object tracker for EdgeGuard AI
# This file exposes the class: SortTracker

import numpy as np


def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in [x1,y1,x2,y2] format.
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    if area_test + area_gt - inter == 0:
        return 0.0

    return inter / (area_test + area_gt - inter)


class KalmanBoxTracker:
    """
    Single object tracker: Kalman filter tracking a bbox.
    """
    count = 0

    def __init__(self, bbox, score=1.0):
        KalmanBoxTracker.count += 1
      
        self.id = KalmanBoxTracker.count
        
        self.last_count = 0
        self.last_conf = 0.0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(8) * 10.0

        self.F = np.eye(8)
        dt = 1.0
        for i in range(4):
            self.F[i, i+4] = dt

        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(8) * 0.01
        self.R = np.eye(4) * 1.0

        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.last_score = float(score)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score=1.0):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)

        z = np.array([cx, cy, w, h], dtype=float)
        y_res = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_res
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_score = float(score)

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return [
            int(cx - w/2),
            int(cy - h/2),
            int(cx + w/2),
            int(cy + h/2)
        ]


class SortTracker:
    """
    SORT multi-object tracker.
    """
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        """
        dets: Nx5 array [x1, y1, x2, y2, score]
        """
        # 1) Predict all trackers
        for t in self.trackers:
            t.predict()

        # 2) Associate dets → tracks
        matches, unmatched_dets, unmatched_tracks = self._associate(dets)

        # 3) Update matched trackers
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(
                dets[det_idx, :4],
                dets[det_idx, 4]
            )

        # 4) Add new trackers
        for det_idx in unmatched_dets:
            bbox = dets[det_idx, :4]
            score = dets[det_idx, 4]
            self.trackers.append(KalmanBoxTracker(bbox, score))

        # 5) Remove aged trackers
        new_tracks = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                new_tracks.append(t)
        self.trackers = new_tracks

        # 6) Output
        outputs = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.age <= self.min_hits:
                outputs.append({
                    "id": t.id,
                    "box": t.get_bbox(),
                    "score": t.last_score
                })

        return outputs

    def _associate(self, dets):
        """
        Greedy IOU matching
        """
        if len(self.trackers) == 0:
            return [], list(range(len(dets))), []

        if len(dets) == 0:
            return [], [], list(range(len(self.trackers)))

        N = len(dets)
        M = len(self.trackers)
        iou_matrix = np.zeros((N, M))

        # Build IoU table
        for d in range(N):
            for t in range(M):
                iou_matrix[d, t] = iou(dets[d, :4], self.trackers[t].get_bbox())


        matches = []
        unmatched_dets = set(range(N))
        unmatched_tracks = set(range(M))

        while True:
            max_idx = np.unravel_index(
                np.argmax(iou_matrix),
                iou_matrix.shape
            )
            d, t = max_idx
            max_iou = iou_matrix[d, t]

            if max_iou < self.iou_threshold:
                break

            matches.append((d, t))
            unmatched_dets.discard(d)
            unmatched_tracks.discard(t)

            iou_matrix[d, :] = -1
            iou_matrix[:, t] = -1

        return matches, list(unmatched_dets), list(unmatched_tracks)
