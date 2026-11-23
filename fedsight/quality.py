"""Inspectable image quality and appearance attributes; no demographic inference."""
from __future__ import annotations
import numpy as np


def crop_components(crop, confidence=1.0) -> dict:
    if crop is None or not hasattr(crop, "shape") or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        return {"valid": False, "score": 0.0, "reason": "empty or malformed crop"}
    from .embeddings import _gray_fixed, crop_quality
    gray = _gray_fixed(crop)
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return {"valid": True, "score": crop_quality(crop, confidence), "sharpness_variance": float(lap.var()),
            "brightness": float(gray.mean() / 255), "contrast": float(gray.std() / 255),
            "height": int(crop.shape[0]), "width": int(crop.shape[1]), "area_px": int(crop.shape[0] * crop.shape[1]),
            "detection_confidence": float(confidence)}


def assess_track(track: dict) -> dict:
    crops = track.get("crops_bgr", [])
    confs = track.get("det_confs", [])
    components = [crop_components(c, confs[i] if i < len(confs) else track.get("avg_conf", 1)) for i, c in enumerate(crops)]
    valid = [c for c in components if c["valid"]]
    return {"components": components, "mean_quality": float(np.mean([c["score"] for c in valid])) if valid else 0.0,
            "track_length": track.get("frame_count", 0), "occlusion_ratio": track.get("occlusion_ratio"),
            "edge_ratio": track.get("edge_ratio"), "coverage": track.get("coverage"),
            "explanation": "Heuristic image usability; not detection accuracy or match probability."}


def appearance_attributes(rgb) -> dict:
    if rgb is None or np.asarray(rgb).size == 0:
        return {"color": "unknown", "method": "unavailable"}
    image = np.asarray(rgb, dtype=float) / 255
    from .attributes import dominant_colour
    h, w = image.shape[:2]
    center = image[h // 4:max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)]
    color = np.median(center.reshape(-1, 3), axis=0)
    palette = {"black": (0.08, .08, .08), "white": (.92, .92, .92), "gray": (.48, .48, .48),
               "red": (.75, .15, .15), "blue": (.15, .25, .7), "green": (.2, .55, .2),
               "yellow": (.85, .8, .2), "brown": (.45, .28, .13)}
    name = min(palette, key=lambda k: np.linalg.norm(color - palette[k]))
    def region_name(top, bottom):
        sample = image[max(0, int(h*top)):max(1, int(h*bottom)), w//4:max(w//4+1, 3*w//4)]
        if not sample.size:
            return "unknown"
        return dominant_colour((sample * 255).astype(np.uint8))
    return {"color": name, "rgb_median": (color * 255).astype(int).tolist(),
            "upper_color": region_name(.2, .5), "lower_color": region_name(.5, .9),
            "method": "dominant quantized HSV region; lighting/background sensitive; unvalidated", "bag": "unknown", "hat": "unknown"}
