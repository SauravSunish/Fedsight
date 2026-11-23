"""Deterministic batch triage on sampled image-plane trajectories."""
from __future__ import annotations
import numpy as np


def point_in_polygon(point, polygon) -> bool:
    """Return whether a normalized point is inside or on a polygon boundary."""
    import cv2
    vertices = np.asarray(polygon, dtype=np.float32)
    return bool(cv2.pointPolygonTest(vertices, (float(point[0]), float(point[1])), False) >= 0)


def line_cross_direction(previous, current, axis: str, position: float) -> str | None:
    """Return the crossing direction on a normalized image axis."""
    index = 0 if axis == "x" else 1
    a, b = previous[index] - position, current[index] - position
    if a < 0 <= b:
        return "positive"
    if a > 0 >= b:
        return "negative"
    return None


def validate_rule(rule: dict):
    if rule["type"] not in {"dwell", "zone", "line", "crowd", "count", "speed", "wrong_direction", "unattended_object"}:
        raise ValueError("Unsupported rule type.")
    if rule["type"] == "speed" and not 0 < rule.get("pixels_per_second", 0) <= 100000:
        raise ValueError("Speed threshold must be positive, in image pixels per second.")
    if rule["type"] == "wrong_direction" and (rule.get("axis") not in {"x", "y"} or rule.get("forbidden_direction") not in {"positive", "negative"}):
        raise ValueError("Direction needs an x/y axis and positive/negative direction.")
    if rule["type"] in {"crowd", "count"} and not 1 <= rule.get("threshold", 0) <= 64:
        raise ValueError("Scene count threshold must be between 1 and 64.")
    if rule["type"] == "crowd" and rule.get("zone_box") is not None:
        box = rule["zone_box"]
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("Crowd zone must be an ordered normalized rectangle.")
    if rule["type"] == "unattended_object" and not (0 < rule.get("seconds", 0) <= 3600 and 0 < rule.get("distance_norm", 0) <= 1):
        raise ValueError("Unattended object requires duration and normalized proximity thresholds.")
    if rule["type"] == "dwell" and not (0 < rule.get("seconds", 0) < 86400):
        raise ValueError("Dwell threshold must be positive and below one day.")
    if rule["type"] == "zone":
        box = rule.get("box", [])
        polygon = rule.get("polygon")
        if polygon is not None:
            if not isinstance(polygon, list) or not 3 <= len(polygon) <= 30 or any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 1 for v in p) for p in polygon):
                raise ValueError("Polygon requires 3–30 normalized [x,y] points.")
            import cv2
            if cv2.contourArea(np.asarray(polygon, dtype=np.float32)) <= 1e-6:
                raise ValueError("Polygon must enclose a nonzero area.")
        elif len(box) != 4 or not all(0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("Zone must be an ordered normalised rectangle or polygon.")
    if rule["type"] == "line":
        if rule.get("axis") not in {"x", "y"} or not 0 < rule.get("position", -1) < 1:
            raise ValueError("Line requires x/y axis and a normalised position.")


def evaluate_rules(track: dict, rules: list[dict], frame_size) -> list[dict]:
    events = []
    width, height = frame_size
    trajectory = np.asarray(track.get("trajectory", []), dtype=float)
    xy = trajectory[:, 1:3] / [width, height] if trajectory.size else np.empty((0, 2))
    for rule in rules:
        validate_rule(rule)
        if rule.get("camera_id") and rule["camera_id"] != track.get("camera_id"):
            continue
        trigger = False
        trigger_frame = None
        if rule["type"] == "dwell":
            trigger = track.get("duration_s", 0) >= rule["seconds"] and track.get("motion", {}).get("stationary", False)
        elif rule["type"] == "speed":
            trigger = track.get("motion", {}).get("speed_px_per_s", 0) > rule["pixels_per_second"]
        elif rule["type"] == "wrong_direction" and len(xy) > 1:
            displacement = xy[-1, 0 if rule["axis"] == "x" else 1] - xy[0, 0 if rule["axis"] == "x" else 1]
            trigger = abs(displacement) >= .02 and ((displacement > 0) == (rule["forbidden_direction"] == "positive"))
        elif rule["type"] == "zone" and len(xy):
            if rule.get("polygon"):
                inside = np.asarray([point_in_polygon(p, rule["polygon"]) for p in xy])
            else:
                x1, y1, x2, y2 = rule["box"]
                inside = (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)
            trigger = bool(inside[0] or ((~inside[:-1]) & inside[1:]).any())
            if trigger:
                trigger_frame = int(trajectory[np.flatnonzero(inside)[0], 0])
        elif rule["type"] == "line" and len(xy) > 1:
            crossings = [(i, line_cross_direction(a, b, rule["axis"], rule["position"])) for i, (a, b) in enumerate(zip(xy[:-1], xy[1:]), 1)]
            valid = [(i, direction) for i, direction in crossings if direction and (not rule.get("direction") or rule["direction"] == direction)]
            trigger = bool(valid)
            if valid:
                trigger_frame = int(trajectory[valid[0][0], 0])
        if trigger:
            start = track["start_time"]
            if trigger_frame is not None and track.get("last_frame", 0) > track.get("first_frame", 0):
                start += ((trigger_frame - track["first_frame"]) / (track["last_frame"] - track["first_frame"])) * (track["end_time"] - track["start_time"])
            events.append({"rule_id": rule["id"], "track_id": track["track_id"], "source_id": track["source_id"],
                           "method": rule["type"], "severity": rule.get("severity", "medium"), "start_time": start,
                           "end_time": start if trigger_frame is not None else track["end_time"], "frame": trigger_frame,
                           "explanation": "Batch rule on sampled detections; review original context. Not a behavioural or criminality label."})
    return events


def evaluate_scene_rules(tracks: list[dict], rules: list[dict]) -> list[dict]:
    """Count distinct sampled person tracklets at each source frame.

    One event per rule/source marks the first observed threshold crossing. This
    is sampled-video triage, not a calibrated count of unique people.
    """
    scene_rules = [rule for rule in rules if rule["type"] in {"crowd", "count"}]
    for rule in scene_rules:
        validate_rule(rule)
    by_frame: dict[tuple[str, int], set[str]] = {}
    positions = {}
    by_track = {t["track_id"]: t for t in tracks}
    for track in tracks:
        if track.get("object_class", "person") != "person":
            continue
        for point in track.get("trajectory", []):
            key = (track["source_id"], int(point[0]))
            by_frame.setdefault(key, set()).add(track["track_id"])
            positions[(track["track_id"], int(point[0]))] = point[1:3]
    events = []
    triggered = set()
    for (source_id, frame), ids in sorted(by_frame.items()):
        first = by_track[min(ids)]
        for rule in scene_rules:
            key = (rule["id"], source_id)
            if key in triggered or (rule.get("camera_id") and rule["camera_id"] != first.get("camera_id")):
                continue
            matching = ids
            if rule["type"] == "crowd" and rule.get("zone_box"):
                x1, y1, x2, y2 = rule["zone_box"]
                matching = {tid for tid in ids if by_track[tid].get("frame_size") and
                            x1 <= positions[(tid, frame)][0] / by_track[tid]["frame_size"][0] <= x2 and
                            y1 <= positions[(tid, frame)][1] / by_track[tid]["frame_size"][1] <= y2}
            if len(matching) >= rule["threshold"]:
                triggered.add(key)
                first = by_track[min(matching)]
                first_frame = first.get("first_frame", frame)
                last_frame = first.get("last_frame", first_frame)
                event_time = (first["start_time"] + (frame - first_frame) *
                              (first["end_time"] - first["start_time"]) / max(1, last_frame - first_frame))
                events.append({"rule_id": rule["id"], "track_id": first["track_id"], "source_id": source_id,
                               "method": rule["type"], "severity": rule.get("severity", "medium"), "frame": frame, "observed_tracklets": len(matching),
                               "start_time": event_time, "end_time": event_time,
                               "explanation": "Threshold observed in a sampled frame; verify against original video."})
    return events + evaluate_unattended(tracks, [r for r in rules if r["type"] == "unattended_object"])


def evaluate_unattended(tracks: list[dict], rules: list[dict]) -> list[dict]:
    """Flag a bag with no nearby sampled person; missing people can cause false flags."""
    for rule in rules:
        validate_rule(rule)
    people = {}
    for track in tracks:
        if track.get("object_class", "person") == "person":
            for p in track.get("trajectory", []):
                if len(p) >= 3:
                    people.setdefault((track["source_id"], int(p[0])), []).append(p[1:3])
    events = []
    for bag in tracks:
        if bag.get("object_class") != "bag" or not bag.get("motion", {}).get("stationary", False) or len(bag.get("trajectory", [])) < 2:
            continue
        width, height = bag.get("frame_size", [0, 0])
        diagonal = float(np.hypot(width, height))
        frame_span = bag.get("last_frame", 0) - bag.get("first_frame", 0)
        if diagonal <= 0 or frame_span <= 0 or bag.get("duration_s", 0) <= 0:
            continue
        seconds_per_frame = bag["duration_s"] / frame_span
        for rule in rules:
            if rule.get("camera_id") and rule["camera_id"] != bag.get("camera_id"):
                continue
            first_unattended = None
            previous_frame = None
            samples = sorted(bag["trajectory"], key=lambda p: p[0])
            frame_gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if b[0] > a[0]]
            typical_gap = float(np.median(frame_gaps)) if frame_gaps else 0
            for point in samples:
                frame = int(point[0])
                # A long sampling gap cannot establish continuous unattended time.
                if previous_frame is not None and frame - previous_frame > max(1, typical_gap * 2):
                    first_unattended = None
                nearby = people.get((bag["source_id"], frame), [])
                is_near = any(np.linalg.norm(np.asarray(xy) - np.asarray(point[1:3])) / diagonal <= rule["distance_norm"] for xy in nearby)
                if is_near:
                    first_unattended = None
                elif first_unattended is None:
                    first_unattended = frame
                elif (frame - first_unattended) * seconds_per_frame >= rule["seconds"]:
                    at = bag["start_time"] + (frame - bag["first_frame"]) * seconds_per_frame
                    events.append({"rule_id": rule["id"], "track_id": bag["track_id"], "source_id": bag["source_id"],
                                   "method": "unattended_object", "severity": rule.get("severity", "medium"),
                                   "frame": frame, "start_frame": first_unattended, "end_frame": frame,
                                   "start_time": at, "end_time": at,
                                   "explanation": "Unattended object candidate based on sampled detections; missed people and tracking gaps can cause false alerts."})
                    break
                previous_frame = frame
    return events


def motion_heatmap(tracks, source_id, frame_size, bins=20):
    width, height = frame_size
    points = [np.asarray(t["trajectory"])[:, 1:3] / [width, height] for t in tracks if t["source_id"] == source_id and len(t.get("trajectory", []))]
    if not points:
        return np.zeros((bins, bins))
    xy = np.concatenate(points)
    return np.histogram2d(xy[:, 1], xy[:, 0], bins=bins, range=[[0, 1], [0, 1]])[0]
