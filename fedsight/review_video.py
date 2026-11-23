"""Sampled tracklet overlays for source-linked video review."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np


def box_at_frame(track: dict[str, Any], frame: int, *, max_gap: int = 40) -> tuple[int, int, int, int] | None:
    """Interpolate nearby sampled boxes; never extrapolate past track endpoints."""
    points = track.get("_render_points")
    if points is None:
        points = sorted((p for p in track.get("trajectory", []) if len(p) >= 5), key=lambda p: p[0])
    if not points:
        return None
    frames = track.get("_render_frames")
    if frames is None:
        frames = [p[0] for p in points]
    index = bisect_left(frames, frame)
    if index < len(points) and frames[index] == frame:
        p = points[index]
    elif index == 0 or index == len(points):
        return None
    else:
        left, right = points[index - 1], points[index]
        if right[0] - left[0] > max_gap:
            return None
        t = (frame - left[0]) / (right[0] - left[0])
        p = [frame] + [left[i] + t * (right[i] - left[i]) for i in range(1, 5)]
    if not all(math.isfinite(float(v)) for v in p[1:5]) or p[3] <= 0 or p[4] <= 0:
        return None
    x, y, width, height = p[1:5]
    return (round(x - width / 2), round(y - height / 2), round(x + width / 2), round(y + height / 2))


def annotate_frame(frame: np.ndarray, tracks: list[dict], frame_number: int, *, max_gap: int = 40,
                   options: dict | None = None, fps: float = 25., time_anchor: dict | None = None,
                   validated_ids: set[str] | None = None, rules: list[dict] | None = None) -> np.ndarray:
    """Draw only supported nearby track boxes onto a copy of the source frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    options = options or {}
    validated_ids = validated_ids or set()
    visible = [(t, box_at_frame(t, frame_number, max_gap=max_gap)) for t in tracks]
    visible = [(t, box) for t, box in visible if box is not None]
    selected = set(options.get("selected_ids", []))
    if options.get("highlight") and selected:
        dark = (output * .3).astype(np.uint8)
        for t, box in visible:
            if t["id"] in selected:
                x1, y1, x2, y2 = box
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 > x1 and y2 > y1:
                    dark[y1:y2, x1:x2] = output[y1:y2, x1:x2]
        output = dark
    zone_alert = False
    if options.get("zones", True):
        for rule in rules or []:
            if rule.get("type") == "zone" and (len(rule.get("box", [])) == 4 or rule.get("polygon")):
                from .rules import point_in_polygon
                if rule.get("polygon"):
                    polygon = rule["polygon"]
                else:
                    x1, y1, x2, y2 = rule["box"]
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                occupied = any(t.get("object_class", "person") == "person" and point_in_polygon(((b[0]+b[2])/(2*width), (b[1]+b[3])/(2*height)), polygon) for t, b in visible)
                zone_alert |= occupied
                pixels = np.asarray([[int(x*width), int(y*height)] for x, y in polygon], dtype=np.int32)
                cv2.polylines(output, [pixels], True, (60, 70, 245) if occupied else (70, 130, 230), 2 if occupied else 1)
            elif rule.get("type") == "line":
                position = rule.get("position", .5)
                a, b = ((int(position * width), 0), (int(position * width), height - 1)) if rule.get("axis") == "x" else ((0, int(position * height)), (width - 1, int(position * height)))
                cv2.line(output, a, b, (210, 180, 70), 1)
    for track, box in visible:
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        color = {"person": (230, 215, 74), "bag": (70, 215, 245), "vehicle": (70, 155, 245)}.get(track.get("object_class", "person"), (230, 215, 74))
        if options.get("trails", True):
            points = track.get("_render_points", track.get("trajectory", []))
            frames = track.get("_render_frames")
            if frames is not None:
                right = bisect_left(frames, frame_number + 1)
                left = max(bisect_left(frames, frame_number - fps * 3), right - 20)
                trail = points[left:right]
            else:
                trail = [p for p in points if len(p) >= 3 and 0 <= frame_number - p[0] <= fps * 3][-20:]
            for a, b in zip(trail, trail[1:]):
                cv2.line(output, (int(a[1]), int(a[2])), (int(b[1]), int(b[2])), color, 1, cv2.LINE_AA)
        if options.get("boxes", True):
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3 if track["id"] in selected and options.get("highlight") else 2)
        if options.get("ids", True):
            label = f"ID {track.get('raw_track_id', track['id'][-6:])} | {track.get('object_class', 'person')} | {track.get('avg_conf', 0):.2f}"
            cv2.putText(output, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    if options.get("hud", True):
        people = sum(t.get("object_class", "person") == "person" for t, _ in visible)
        bags = sum(t.get("object_class") == "bag" for t, _ in visible)
        crowd = people >= int(options.get("crowd_threshold", 5))
        cards = [("PEOPLE TRACKED", str(people), (230, 215, 74)),
                 ("CROWD", "ALERT" if crowd else "OK", (75, 75, 245) if crowd else (120, 220, 150)),
                 ("LINK REVIEW", "VALIDATED" if any(t["id"] in validated_ids for t, _ in visible) else "TRACKING", (120, 220, 150)),
                 ("OBJECTS", f"BAGS {bags}", (70, 215, 245))]
        if zone_alert:
            cards.append(("ZONE FLAG", "PRESENCE", (75, 75, 245)))
        for index, (title, value, color) in enumerate(cards):
            top = 10 + index * 42
            if top + 38 >= height:
                break
            right = min(width - 1, 190)
            overlay = output.copy()
            cv2.rectangle(overlay, (8, top), (right, top + 36), (18, 27, 35), -1)
            cv2.addWeighted(overlay, .8, output, .2, 0, output)
            cv2.putText(output, title, (14, top + 13), cv2.FONT_HERSHEY_SIMPLEX, .32, (195, 205, 210), 1, cv2.LINE_AA)
            cv2.putText(output, value, (14, top + 29), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    if options.get("timestamp", True):
        time_label = f"CLIP {frame_number / max(fps, .001):.2f}s | FRAME {frame_number}"
        if time_anchor:
            from .core import TimeAnchor
            anchor = TimeAnchor(**time_anchor)
            bounds = anchor.interval(frame_number / max(fps, .001))
            if bounds:
                time_label += " | UTC " + datetime.fromtimestamp(sum(bounds)/2, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + f" +/-{anchor.uncertainty_s:g}s"
        cv2.putText(output, time_label, (8, max(15, height - 10)), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def track_density(tracks: list[dict], duration_s: float, *, bins: int = 40) -> list[dict[str, float]]:
    """Count tracklet intervals intersecting each clip-relative time bin."""
    if duration_s <= 0 or bins < 1:
        return []
    result = []
    for index in range(bins):
        start, end = duration_s * index / bins, duration_s * (index + 1) / bins
        count = sum(t.get("start_time", 0) < end and t.get("end_time", 0) >= start for t in tracks)
        result.append({"clip_seconds": round(start, 2), "tracklets": count})
    return result


def similar_tracks(query: dict, tracks: list[dict], *, limit: int = 10) -> list[tuple[dict, float]]:
    """Rank compatible indexed descriptors by cosine; scores are not probabilities."""
    if query.get("embedding") is None:
        return []
    vector = np.asarray(query["embedding"], dtype=float).reshape(-1)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-10:
        return []
    vector /= np.linalg.norm(vector)
    matches = []
    for track in tracks:
        if track["id"] == query["id"] or track.get("object_class", "person") != query.get("object_class", "person") or track.get("embedding") is None:
            continue
        other = np.asarray(track["embedding"], dtype=float).reshape(-1)
        if other.shape != vector.shape or not np.isfinite(other).all() or np.linalg.norm(other) < 1e-10:
            continue
        matches.append((track, float(vector @ (other / np.linalg.norm(other)))))
    return sorted(matches, key=lambda row: (-row[1], row[0]["id"]))[:max(0, limit)]


def render_annotated_clip(input_path: str, output_path: str, tracks: list[dict], *, start_s: float, duration_s: float,
                          options: dict | None = None, time_anchor: dict | None = None,
                          validated_ids: set[str] | None = None, rules: list[dict] | None = None) -> dict:
    """Render a bounded silent MP4 with source-relative time and sampled overlays."""
    if not all(math.isfinite(x) for x in (start_s, duration_s)) or start_s < 0 or not 0 < duration_s <= 300:
        raise ValueError("Choose a nonnegative start and a duration up to 5 minutes.")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Original recording could not be decoded.")
    writer = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or not 1 <= fps <= 120 or min(width, height) < 2:
            raise ValueError("Recording needs valid frame dimensions and FPS for annotated export.")
        scale = min(1., 1280 / max(width, height))
        out_width = max(2, int(width * scale) // 2 * 2)
        out_height = max(2, int(height * scale) // 2 * 2)
        start_frame = int(start_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
        if not writer.isOpened():
            raise OSError("MP4 encoder is unavailable on this workstation.")
        count = 0
        prepared_tracks = []
        for track in tracks:
            points = sorted((p for p in track.get("trajectory", []) if len(p) >= 5), key=lambda p: p[0])
            prepared_tracks.append({**track, "_render_points": points, "_render_frames": [p[0] for p in points]})
        for _ in range(int(math.ceil(duration_s * fps))):
            ok, frame = cap.read()
            if not ok:
                break
            actual_frame = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            image = annotate_frame(frame, prepared_tracks, actual_frame, options=options, fps=fps,
                                   time_anchor=time_anchor, validated_ids=validated_ids, rules=rules)
            if (out_width, out_height) != (width, height):
                image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            count += 1
        if not count:
            raise ValueError("No video frames were found at the selected time.")
        writer.release()
        writer = None
        codec = "mp4v"
        executable = shutil.which("ffmpeg")
        if not executable:
            try:
                import imageio_ffmpeg
                executable = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                pass
        if executable:
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(output_path) or ".", suffix=".mp4", delete=False) as temp:
                converted = temp.name
            try:
                result = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", "-i", output_path,
                                         "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                                         "-movflags", "+faststart", converted], capture_output=True, timeout=600, check=False)
                if result.returncode == 0 and os.path.getsize(converted) > 0:
                    os.replace(converted, output_path)
                    codec = "h264"
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                if os.path.exists(converted):
                    os.unlink(converted)
        return {"start_frame": start_frame, "frames": count, "fps": fps, "duration_s": count / fps,
                "frame_size": [out_width, out_height], "audio": False, "codec": codec,
                "overlay": "Sampled boxes, linearly interpolated only across short gaps."}
    finally:
        if writer is not None:
            writer.release()
        cap.release()
