"""Within-video motion/IoU association. No native LAP extension required."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class State:
    box: np.ndarray
    velocity: np.ndarray
    last_seen: int


class MotionIoUTracker:
    """Track IDs local to one clip. Short gaps are predicted at constant velocity."""
    def __init__(self, min_iou=.25, max_missed=6):
        self.min_iou, self.max_missed = min_iou, max_missed
        self.active = {}
        self.step, self.next_id = 0, 1

    def update(self, boxes):
        self.step += 1
        self.active = {i: s for i, s in self.active.items() if self.step - s.last_seen <= self.max_missed + 1}
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        ids = np.full(len(boxes), -1, dtype=int)
        prior = list(self.active)
        if prior and len(boxes):
            predicted = np.stack([s.box + s.velocity * (self.step - s.last_seen) for s in self.active.values()])
            low = np.maximum(predicted[:, None, :2], boxes[None, :, :2])
            high = np.minimum(predicted[:, None, 2:], boxes[None, :, 2:])
            intersection = np.maximum(high - low, 0).prod(axis=2)
            area_p = np.maximum(predicted[:, 2:] - predicted[:, :2], 0).prod(axis=1)
            area_b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
            iou = intersection / np.maximum(area_p[:, None] + area_b - intersection, 1e-9)
            rows, cols = linear_sum_assignment(np.where(iou >= self.min_iou, 1. - iou, 1e6))
            for r, c in zip(rows, cols):
                if iou[r, c] < self.min_iou:
                    continue
                tid, box = prior[r], boxes[c]
                state = self.active[tid]
                observed = (box - state.box) / (self.step - state.last_seen)
                self.active[tid] = State(box.copy(), .5 * state.velocity + .5 * observed, self.step)
                ids[c] = tid
        for index in np.flatnonzero(ids < 0):
            tid = self.next_id
            self.next_id += 1
            self.active[tid] = State(boxes[index].copy(), np.zeros(4), self.step)
            ids[index] = tid
        return ids


def resolve_tracker(requested):
    if requested == "portable-iou":
        return "portable-iou", "Portable motion/IoU assignment selected."
    try:
        import lap
        if not callable(getattr(lap, "lapjv", None)):
            raise ImportError("LAP assignment is unavailable.")
    except (ImportError, OSError) as exc:
        if requested != "auto":
            raise ValueError(f"{requested} requires the unavailable LAP extension. Select Auto or Portable motion/IoU. Details: {exc}") from exc
        return "portable-iou", "Native LAP unavailable; Auto selected portable motion/IoU."
    return ("bytetrack.yaml" if requested == "auto" else requested), "Native tracker available."
