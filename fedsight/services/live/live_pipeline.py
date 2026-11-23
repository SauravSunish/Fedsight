"""Incremental person tracking and deterministic live rule evaluation."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from ...core import uid
from ...portable_tracker import MotionIoUTracker
from ...quality import appearance_attributes
from ...review_video import annotate_frame
from ...rules import line_cross_direction, point_in_polygon
from .archive import utc


class Detector:
    def __init__(self):
        self.lock = Lock()
        self.model = None

    def __call__(self, frame):
        with self.lock:
            if self.model is None:
                from ultralytics import YOLO
                weights = Path(__file__).resolve().parents[3] / "yolo11n.pt"
                if not weights.is_file():
                    raise ValueError("Local yolo11n.pt weights are missing.")
                self.model = YOLO(str(weights))
            result = self.model.predict(frame, classes=[0], conf=.35, max_det=64, imgsz=480,
                                        device="cpu", verbose=False)[0].boxes
            return result.xyxy.cpu().numpy(), result.conf.cpu().numpy()


def protected(frame, mask):
    if mask:
        # Mask the entire scene so missed detections cannot expose a face.
        tiny = cv2.resize(frame, (12, 8), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return frame


def inside(point, rule):
    if rule.get("polygon"):
        return point_in_polygon(point, rule["polygon"])
    x1,y1,x2,y2 = rule["box"]
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def matched(rule, tracks, previous, stamp, camera):
    if rule.get("camera_id") and rule["camera_id"] != camera:
        return []
    hour = datetime.fromtimestamp(stamp, timezone.utc).hour
    start, end = rule.get("hours", [0,24])
    if not (start <= hour < end if start < end else hour >= start or hour < end):
        return []
    kind = rule["type"]
    if kind in ("crowd","count"):
        return [(None, None)] if len(tracks) >= rule["threshold"] else []
    hits = []
    for t in tracks:
        tid, pt = t["id"], t["center"]
        old = previous.get(tid)
        direction = None
        if kind == "line":
            direction = line_cross_direction(old, pt, rule["axis"], rule["position"]) if old else None
            hit = bool(direction and direction == (rule.get("direction") or direction))
        elif kind == "zone":
            hit = inside(pt, rule) and (old is None or not inside(old, rule))
        elif kind == "appearance":
            hit = t["attributes"].get("upper_color") == rule["upper_color"]
        else:
            continue
        if hit:
            hits.append((tid, direction))
    return hits


class Cooldown:
    def __init__(self, seconds=30):
        self.seconds = seconds
        self.last = {}

    def accept(self, rule, track, stamp):
        key = (rule, track)
        self.last = {k:t for k,t in self.last.items() if stamp-t < self.seconds}
        if key in self.last:
            return False
        self.last[key] = stamp
        return True


class Pipeline:
    def __init__(self, detector):
        self.detector = detector
        self.run = uid("stream")
        self.tracker = MotionIoUTracker()
        self.history = {}
        self.previous = {}
        self.cooldown = Cooldown()
        self.index = 0

    def step(self, raw, rules, camera, stamp):
        boxes, confidence = self.detector(raw)
        ids = self.tracker.update(boxes)
        h,w = raw.shape[:2]
        tracks = []
        self.index += 1
        for bbox, ident, conf in zip(boxes,ids,confidence):
            x1,y1 = np.maximum(bbox[:2], 0).astype(int)
            x2,y2 = np.minimum(bbox[2:], [w,h]).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            x,y = float(x1+x2)/2,float(y1+y2)/2
            tid = f"{self.run}_{int(ident)}"
            trail = self.history.setdefault(tid, deque(maxlen=12))
            trail.append([self.index,x,y,int(x2-x1),int(y2-y1)])
            crop = raw[y1:y2,x1:x2,::-1]
            tracks.append(dict(id=tid, raw_track_id=int(ident), object_class="person", center=[x/w,y/h],
                               avg_conf=float(conf), trajectory=list(trail), attributes=appearance_attributes(crop)))
        self.history = {t["id"]: self.history[t["id"]] for t in tracks}
        hits = []
        for rule in rules:
            for tid, direction in matched(rule,tracks,self.previous,stamp,camera):
                if self.cooldown.accept(rule["id"],tid,stamp):
                    hits.append(dict(rule_id=rule["id"], rule=rule["label"], method=rule["type"],
                                     severity=rule["severity"], track_id=tid, direction=direction))
        self.previous = {t["id"]:t["center"] for t in tracks}
        return tracks,hits


def rendered(frame, tracks, rules, stamp, options=None):
    opts = {**(options or {}), "timestamp": False}
    index = tracks[0]["trajectory"][-1][0] if tracks else 0
    view = annotate_frame(frame, tracks, index, options=opts, rules=rules, fps=3)
    cv2.putText(view, utc(stamp), (8, view.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (255,255,255), 1, cv2.LINE_AA)
    return view
