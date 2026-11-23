"""Approximate, deterministic HSV colour estimates for tracklet crops."""
from __future__ import annotations

import cv2
import numpy as np


def dominant_colour(rgb: np.ndarray) -> str:
    """Find the most populated quantized HSV cluster; lighting affects results."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        return "unknown"
    small = cv2.resize(image.astype(np.uint8), (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    labels = np.full(len(hsv), "unknown", dtype="<U8")
    labels[v < 50] = "black"
    labels[(v >= 50) & (s < 45) & (v >= 200)] = "white"
    labels[(v >= 50) & (s < 45) & (v < 200)] = "gray"
    chromatic = (v >= 50) & (s >= 45)
    labels[chromatic & ((h < 10) | (h >= 170))] = "red"
    labels[chromatic & (h >= 10) & (h < 25)] = "brown"
    labels[chromatic & (h >= 25) & (h < 40)] = "yellow"
    labels[chromatic & (h >= 40) & (h < 90)] = "green"
    labels[chromatic & (h >= 90) & (h < 140)] = "blue"
    labels[chromatic & (h >= 140) & (h < 170)] = "red"
    values, counts = np.unique(labels, return_counts=True)
    return str(values[int(np.argmax(counts))])
