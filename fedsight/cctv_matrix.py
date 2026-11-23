"""Multi-Camera CCTV Video Matrix ("Wall of Eyes") for synchronized tactical playback."""
from __future__ import annotations

import io
import math
import tempfile
import os
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2

from .core import TimeAnchor


def get_active_tracks_at_time(tracks: list[dict], source_id: str, query_time_s: float) -> list[dict]:
    """Return all tracklets active at query_time_s for the given source."""
    active = []
    for t in tracks:
        if t.get("source_id") != source_id:
            continue
        start = float(t.get("start_time", 0.0))
        end = float(t.get("end_time", start))
        if start <= query_time_s <= end:
            active.append(t)
    return active


def interpolate_track_box(track: dict, query_time_s: float, fps: float = 25.0,
                          frame_w: int = 640, frame_h: int = 360) -> tuple[int, int, int, int, float]:
    """Interpolate (x, y, w, h, speed_px_s) for a track at query_time_s."""
    traj = track.get("trajectory")
    start = float(track.get("start_time", 0.0))
    duration = max(0.1, float(track.get("end_time", start) - start))
    progress = max(0.0, min(1.0, (query_time_s - start) / duration))

    if traj is not None and len(traj) > 0:
        arr = np.asarray(traj, dtype=float)
        # cols: frame, x, y, w, h
        target_frame = query_time_s * fps
        frames = arr[:, 0]
        idx = int(np.clip(np.searchsorted(frames, target_frame), 0, len(arr) - 1))
        row = arr[idx]
        x, y = int(row[1]), int(row[2])
        w = int(row[3]) if len(row) > 3 else 50
        h = int(row[4]) if len(row) > 4 else 100

        # Calculate instantaneous speed
        if idx > 0:
            dt = max(0.04, (frames[idx] - frames[idx - 1]) / fps)
            dist = math.hypot(arr[idx, 1] - arr[idx - 1, 1], arr[idx, 2] - arr[idx - 1, 2])
            speed = dist / dt
        else:
            speed = 35.0
        return x, y, w, h, speed

    # Fallback interpolation if no trajectory table
    x = int(80 + progress * (frame_w - 180))
    y = int(frame_h * 0.45)
    w, h = 55, 110
    speed = 42.0
    return x, y, w, h, speed


def draw_cctv_frame_overlays(img: Image.Image, camera_title: str, query_time_s: float,
                             active_targets: list[dict], time_anchor: TimeAnchor | None = None,
                             fps: float = 25.0) -> Image.Image:
    """Burn in tactical CCTV metadata (REC icon, green timecode, camera header, targeting brackets)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Timecode calculation
    if time_anchor and time_anchor.original_timestamp:
        bounds = time_anchor.interval(query_time_s)
        if bounds:
            import pandas as pd
            instant = pd.to_datetime(sum(bounds) / 2, unit="s", utc=True)
            time_str = instant.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            time_str = f"T+{query_time_s:06.2f}s"
    else:
        time_str = f"T+{query_time_s:06.2f}s"

    # Top banner bar (Semi-transparent black)
    draw.rectangle([(0, 0), (w, 32)], fill=(4, 9, 14, 210))

    # Camera label (Top Left)
    cam_text = camera_title.upper()
    draw.text((12, 8), cam_text, fill=(80, 226, 217))

    # REC & Timecode (Top Right)
    # Draw red flashing REC circle
    draw.ellipse([(w - 220, 11), (w - 210, 21)], fill=(239, 68, 68))
    rec_text = f"DEMO  {time_str}"
    draw.text((w - 202, 8), rec_text, fill=(74, 222, 128))

    # Bottom telemetry bar
    draw.rectangle([(0, h - 24), (w, h)], fill=(4, 9, 14, 190))
    tgt_count = len(active_targets)
    tgt_status = f"ACTIVE TARGETS: {tgt_count:02d}" if tgt_count > 0 else "NO ACTIVE TARGET"
    draw.text((12, h - 18), tgt_status, fill=(80, 226, 217) if tgt_count > 0 else (120, 144, 156))
    draw.text((w - 140, h - 18), "Demo illustration", fill=(100, 125, 140))

    # Draw targeting brackets for active targets
    if active_targets:
        for track in active_targets:
            bx, by, bw, bh, speed = interpolate_track_box(track, query_time_s, fps=fps, frame_w=w, frame_h=h)
            # Center coordinates to bounding box
            x1 = max(4, bx - bw // 2)
            y1 = max(34, by - bh // 2)
            x2 = min(w - 4, x1 + bw)
            y2 = min(h - 26, y1 + bh)

            # High-tech targeting corner brackets
            bracket_color = (80, 226, 217)
            bracket_len = min(12, bw // 3, bh // 3)
            # Top-left
            draw.line([(x1, y1), (x1 + bracket_len, y1)], fill=bracket_color, width=2)
            draw.line([(x1, y1), (x1, y1 + bracket_len)], fill=bracket_color, width=2)
            # Top-right
            draw.line([(x2, y1), (x2 - bracket_len, y1)], fill=bracket_color, width=2)
            draw.line([(x2, y1), (x2, y1 + bracket_len)], fill=bracket_color, width=2)
            # Bottom-left
            draw.line([(x1, y2), (x1 + bracket_len, y2)], fill=bracket_color, width=2)
            draw.line([(x1, y2), (x1, y2 - bracket_len)], fill=bracket_color, width=2)
            # Bottom-right
            draw.line([(x2, y2), (x2 - bracket_len, y2)], fill=bracket_color, width=2)
            draw.line([(x2, y2), (x2, y2 - bracket_len)], fill=bracket_color, width=2)

            # Target tag
            tid = track["id"][-6:].upper()
            tag_text = f"Track {tid}"
            # Background pill for tag
            draw.rectangle([(x1, max(34, y1 - 18)), (x1 + 95, max(34, y1 - 2))], fill=(8, 22, 32))
            draw.text((x1 + 3, max(36, y1 - 16)), tag_text, fill=(80, 226, 217))
    else:
        # Subtle watermark when no target is present
        no_tgt = "[ NO ACTIVE TARGET ]"
        draw.text((w // 2 - 65, h // 2 - 8), no_tgt, fill=(45, 68, 85))

    return img


def generate_synthetic_cctv_frame(camera_title: str, query_time_s: float, active_targets: list[dict],
                                  time_anchor: TimeAnchor | None = None, fps: float = 25.0,
                                  width: int = 540, height: int = 300) -> Image.Image:
    """Generate a realistic synthetic CCTV video frame with room perspective and targets."""
    # Create dark surveillance room canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Ambient dark slate blue
    canvas[:, :] = [10, 18, 26]

    # Draw architectural perspective grid lines (corridor / room walls)
    cv2.line(canvas, (0, 0), (int(width * 0.28), int(height * 0.35)), (22, 38, 52), 1)
    cv2.line(canvas, (width, 0), (int(width * 0.72), int(height * 0.35)), (22, 38, 52), 1)
    cv2.line(canvas, (0, height), (int(width * 0.28), int(height * 0.65)), (22, 38, 52), 1)
    cv2.line(canvas, (width, height), (int(width * 0.72), int(height * 0.65)), (22, 38, 52), 1)
    # Horizon / back wall rectangle
    cv2.rectangle(canvas, (int(width * 0.28), int(height * 0.35)), (int(width * 0.72), int(height * 0.65)), (20, 35, 48), 1)

    # Subtle floor tile grid
    for floor_y in range(int(height * 0.65), height, 28):
        cv2.line(canvas, (0, floor_y), (width, floor_y), (14, 26, 36), 1)

    # Scanline effect
    for y in range(0, height, 4):
        canvas[y, :] = (canvas[y, :].astype(float) * 0.88).astype(np.uint8)

    # Render active subjects if present
    for track in active_targets:
        bx, by, bw, bh, _ = interpolate_track_box(track, query_time_s, fps=fps, frame_w=width, frame_h=height)
        # Render a silhouette body
        color_attr = track.get("attributes", {}).get("color", "blue").lower()
        palette_map = {
            "blue": (180, 100, 35),
            "red": (45, 45, 190),
            "green": (45, 160, 60),
            "yellow": (40, 180, 210),
            "black": (30, 35, 42),
            "white": (200, 205, 210),
        }
        b_color = palette_map.get(color_attr, (120, 130, 140))
        # Head
        head_r = max(6, bw // 5)
        cv2.circle(canvas, (bx, max(38, by - bh // 2 + head_r)), head_r, (150, 160, 175), -1)
        # Torso
        torso_top = max(42, by - bh // 2 + head_r * 2)
        torso_bottom = min(height - 30, torso_top + int(bh * 0.45))
        cv2.rectangle(canvas, (bx - bw // 3, torso_top), (bx + bw // 3, torso_bottom), b_color, -1)
        # Legs
        cv2.rectangle(canvas, (bx - bw // 3, torso_bottom), (bx - 2, min(height - 26, torso_bottom + int(bh * 0.4))), (40, 50, 65), -1)
        cv2.rectangle(canvas, (bx + 2, torso_bottom), (bx + bw // 3, min(height - 26, torso_bottom + int(bh * 0.4))), (40, 50, 65), -1)

    pil_img = Image.fromarray(canvas)
    return draw_cctv_frame_overlays(pil_img, camera_title, query_time_s, active_targets, time_anchor=time_anchor, fps=fps)


def extract_or_render_camera_frame(service: Any, source: dict, camera_title: str,
                                   active_targets: list[dict], query_time_s: float,
                                   actor: str = "reviewer", fps: float = 25.0,
                                   width: int = 540, height: int = 300) -> Image.Image:
    """Return source-linked imagery. Only explicit demo sources get illustrations."""
    if source.get("demo") and not source.get("asset"):
        anchor = TimeAnchor(**(source.get("time_anchor") or {}))
        return generate_synthetic_cctv_frame(camera_title, query_time_s, active_targets, anchor, fps, width, height)
    if not source.get("asset"):
        raise ValueError("Original video unavailable.")
    from .review_video import annotate_frame
    blob = service.read_asset(source["asset"], actor)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=service.work, suffix=source.get("suffix", ".mp4"), delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(blob)
        cap = cv2.VideoCapture(str(tmp_path))
        try:
            recorded_fps = cap.get(cv2.CAP_PROP_FPS)
            if recorded_fps > 0 and math.isfinite(recorded_fps):
                fps = recorded_fps
            frame_no = int(query_time_s * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, bgr = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError("No readable frame at this offset.")
        annotated = annotate_frame(bgr, active_targets, frame_no, fps=fps, time_anchor=source.get("time_anchor"),
                                   options=dict(hud=False, trails=False, timestamp=False))
        frame = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        tile = ImageOps.pad(frame, (width, height), color="#09111b")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, width, 22), fill="#09111b")
        draw.text((7, 4), f"{camera_title}  |  +{query_time_s:.1f}s", fill="#e7f2fa")
        return tile
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def compose_cctv_matrix(frames: list[tuple[str, Image.Image]], grid: tuple[int, int] = (2, 2)) -> Image.Image:
    """Compose up to N camera frames into a unified 2x2 CCTV Control Wall matrix image."""
    cols, rows = grid
    if not frames:
        return Image.new("RGB", (640, 360), (10, 15, 20))

    frame_w, frame_h = frames[0][1].size
    gutter = 6
    total_w = cols * frame_w + (cols - 1) * gutter
    total_h = rows * frame_h + (rows - 1) * gutter

    matrix_img = Image.new("RGB", (total_w, total_h), (5, 9, 14))

    for idx, (cam_id, frame) in enumerate(frames[: cols * rows]):
        r = idx // cols
        c = idx % cols
        x = c * (frame_w + gutter)
        y = r * (frame_h + gutter)
        matrix_img.paste(frame, (x, y))

    return matrix_img
