"""Simple vehicle appearance retrieval baseline; no identity training."""
from __future__ import annotations

import cv2
import numpy as np


def vehicle_descriptor(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 2:
        raise ValueError("A valid BGR vehicle crop is required.")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Spatial cells retain a little layout information while limiting lighting sensitivity.
    parts = []
    for cell in (hsv, hsv[:len(hsv)//2], hsv[len(hsv)//2:]):
        for channel, bins, limit in ((0, 16, 180), (1, 8, 256), (2, 8, 256)):
            hist = cv2.calcHist([cell], [channel], None, [bins], [0, limit]).ravel()
            parts.append(hist / max(float(hist.sum()), 1.0))
    descriptor = np.concatenate(parts).astype(np.float32)
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)


def describe_vehicles(tracks: list[dict]) -> list[dict]:
    for track in tracks:
        crops = track.get("crops_bgr", [])
        vectors = [vehicle_descriptor(c) for c in crops if c is not None and min(c.shape[:2]) >= 2]
        if not vectors:
            continue
        gallery = np.stack(vectors)
        mean = gallery.mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-8)
        track["embeddings"] = gallery.tolist()
        track["embedding"] = mean.tolist()
        track["quality"] = [1.0] * len(vectors)
        track["embedding_meta"] = {"method": "HSV colour histogram", "trained_reid": False,
                                   "mean_quality": 0.5, "possible_id_switch": False}
    return tracks
