"""Time-ordered review reel from indexed track previews."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def review_reel(service, tracks: list[dict], sources: dict, actor: str, *, limit: int = 60) -> bytes:
    selected = sorted(tracks, key=lambda t: (sources[t["source_id"]]["name"], t["start_time"], t["track_id"]))[:limit]
    if not selected:
        raise ValueError("No tracklets match the current search.")
    with tempfile.NamedTemporaryFile(dir=service.work, suffix=".mp4", delete=False) as tmp:
        path = Path(tmp.name)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (960, 540))
    if not writer.isOpened():
        path.unlink(missing_ok=True)
        raise OSError("MP4 encoder is unavailable on this workstation.")
    try:
        for index, track in enumerate(selected, 1):
            asset = track.get("preview_asset")
            if not asset:
                continue
            image = cv2.imdecode(np.frombuffer(service.read_asset(asset, actor), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            canvas = np.full((540, 960, 3), (38, 32, 24), dtype=np.uint8)
            h, w = image.shape[:2]
            scale = min(600 / max(w, 1), 420 / max(h, 1))
            resized = cv2.resize(image, (max(1, int(w*scale)), max(1, int(h*scale))))
            y, x = (540-resized.shape[0])//2, (960-resized.shape[1])//2
            canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            source = sources[track["source_id"]]["name"][:54]
            lines = [f"REVIEW REEL  {index}/{len(selected)}", source,
                     f"Clip time  {track['start_time']:.1f}s - {track['end_time']:.1f}s",
                     f"{track.get('object_class', 'person').title()} tracklet  {track['track_id'][-12:]}"]
            for line_no, line in enumerate(lines):
                cv2.putText(canvas, line, (24, 32 + 30*line_no), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            for _ in range(3):
                writer.write(canvas)
    except BaseException:
        writer.release()
        path.unlink(missing_ok=True)
        raise
    writer.release()
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise OSError("No review reel frames could be encoded.")
    output = path.read_bytes()
    path.unlink(missing_ok=True)
    return output
