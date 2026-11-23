"""Local webcam monitoring with bounded, session-scoped alerts."""
from __future__ import annotations

from datetime import datetime, timezone
import time

import cv2
import numpy as np

from .portable_tracker import MotionIoUTracker


class LiveSession:
    def __init__(self, camera_index: int, *, confidence: float = .35):
        if camera_index not in (0, 1):
            raise ValueError("Only local cameras 0 and 1 are supported.")
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Local camera {camera_index} could not be opened.")
        self.camera_index = camera_index
        self.confidence = confidence
        self.tracker = MotionIoUTracker()
        self.previous: dict[int, tuple[float, float]] = {}
        self.alerts: list[dict] = []
        self.last_alert: dict[str, float] = {}
        self.started = time.monotonic()
        self.closed = False

    def close(self):
        if not self.closed:
            self.capture.release()
            self.closed = True

    def _alert(self, rule: dict, count: int, ids: list[int]):
        current = time.monotonic()
        if current - self.last_alert.get(rule["id"], -1e9) < 10:
            return None
        self.last_alert[rule["id"]] = current
        event = {"time_utc": datetime.now(timezone.utc).isoformat(), "rule": rule.get("label", rule["type"]),
                 "type": rule["type"], "count": count, "track_ids": ",".join(map(str, ids)),
                 "camera": self.camera_index, "severity": rule.get("severity", "medium")}
        self.alerts.append(event)
        self.alerts = self.alerts[-200:]
        return event

    def step(self, model, rules: list[dict]):
        if self.closed or time.monotonic() - self.started > 3600:
            self.close()
            raise ValueError("Live session stopped after one hour. Start a new session to continue.")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.close()
            raise ValueError("Camera frame is unavailable.")
        result = model.predict(frame, classes=[0], conf=self.confidence, max_det=64,
                               imgsz=640, verbose=False, device="cpu")[0]
        raw = result.boxes
        boxes = raw.xyxy.cpu().numpy() if raw is not None else np.empty((0, 4))
        ids = self.tracker.update(boxes)
        height, width = frame.shape[:2]
        points = {int(tid): ((box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height))
                  for tid, box in zip(ids, boxes)}
        new_alerts = []
        for rule in rules:
            if rule["type"] not in {"zone", "line", "crowd", "count"}:
                continue
            matching = []
            if rule["type"] in {"crowd", "count"}:
                if len(points) >= rule["threshold"]:
                    matching = list(points)
            elif rule["type"] == "zone":
                from .rules import point_in_polygon
                if rule.get("polygon"):
                    polygon = rule["polygon"]
                else:
                    x1, y1, x2, y2 = rule["box"]
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                matching = [tid for tid, point in points.items() if point_in_polygon(point, polygon)
                            and (tid not in self.previous or not point_in_polygon(self.previous[tid], polygon))]
                pixels = np.asarray([[int(x*width), int(y*height)] for x, y in polygon], dtype=np.int32)
                cv2.polylines(frame, [pixels], True, (0, 180, 220), 2)
            else:
                axis, boundary = (0 if rule["axis"] == "x" else 1), rule["position"]
                matching = [tid for tid, point in points.items() if tid in self.previous
                            and (point[axis] - boundary) * (self.previous[tid][axis] - boundary) < 0]
                if axis == 0:
                    cv2.line(frame, (int(boundary*width), 0), (int(boundary*width), height), (220, 160, 0), 2)
                else:
                    cv2.line(frame, (0, int(boundary*height)), (width, int(boundary*height)), (220, 160, 0), 2)
            if matching:
                alert = self._alert(rule, len(points), matching)
                if alert:
                    new_alerts.append(alert)
        self.previous = points
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (42, 196, 132), 2)
            cv2.putText(frame, f"person {tid}", (x1, max(y1-6, 14)), cv2.FONT_HERSHEY_SIMPLEX, .5, (42, 196, 132), 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), len(points), new_alerts
