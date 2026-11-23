"""Bounded, local-only person tracking with Re-ID-ready tracklets.

This module treats video as untrusted input: it limits metadata, workload,
detector/tracker choices, and retained objects before calling the model. It is
still a consented single-user prototype; sandbox it before accepting untrusted media.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class VideoSafetyError(RuntimeError):
    """Raised when a video or processing configuration exceeds local limits."""


@dataclass(frozen=True)
class AnalyzeConfig:
    """Validated policy for a single local analysis job."""

    weights: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    confidence: float = 0.25
    classes: tuple[int, ...] = (0,)
    iou: float | None = None
    imgsz: int = 640
    device: str | None = None
    half: bool = True
    frame_step: int = 2
    max_processed_frames: int = 800
    max_crops: int = 8
    min_crop_height: int = 48
    min_crop_width: int = 24
    min_aspect: float = 0.8
    max_aspect: float = 6.0
    crop_pad: float = 0.03
    min_crop_gap_frames: int | None = None
    min_track_frames: int = 2
    min_avg_conf: float = 0.0
    max_trajectory_points: int = 2000
    max_tracks: int = 128
    max_detections_per_frame: int = 64
    max_wall_time_s: float = 600.0
    max_file_bytes: int = 200 * 1024 * 1024
    max_total_frames: int = 30_000
    max_duration_s: float = 300.0
    max_width: int = 3840
    max_height: int = 2160
    default_fps: float = 25.0
    drop_stationary: bool = False
    stationary_motion_frac: float = 0.03

    def validate(self) -> None:
        if self.weights not in {"yolo11n.pt", "yolo26n.pt", "yolov8n.pt"}:
            raise VideoSafetyError("Detector weights are not approved for this local build.")
        if self.tracker not in {"auto", "portable-iou", "bytetrack.yaml", "botsort.yaml"}:
            raise VideoSafetyError("Tracker configuration is not approved for this local build.")
        if not self.classes or not set(self.classes) <= {0, 1, 2, 3, 5, 7, 24, 26, 28}:
            raise VideoSafetyError("Choose supported person and vehicle detector classes.")
        if not 0.0 < self.confidence <= 1.0:
            raise VideoSafetyError("Detection confidence must be between 0 and 1.")
        if self.iou is not None and not 0.0 < self.iou <= 1.0:
            raise VideoSafetyError("NMS IoU must be between 0 and 1.")
        if not 256 <= self.imgsz <= 2048 or self.imgsz % 32:
            raise VideoSafetyError("Image size must be a multiple of 32 between 256 and 2048.")
        positive = {
            "frame_step": self.frame_step, "max_processed_frames": self.max_processed_frames,
            "max_crops": self.max_crops, "min_crop_height": self.min_crop_height,
            "min_crop_width": self.min_crop_width, "min_track_frames": self.min_track_frames,
            "max_trajectory_points": self.max_trajectory_points, "max_tracks": self.max_tracks,
            "max_detections_per_frame": self.max_detections_per_frame,
            "max_file_bytes": self.max_file_bytes, "max_total_frames": self.max_total_frames,
            "max_width": self.max_width, "max_height": self.max_height,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise VideoSafetyError(f"Processing limits must be positive: {', '.join(invalid)}.")
        if self.min_crop_gap_frames is not None and self.min_crop_gap_frames < 1:
            raise VideoSafetyError("Minimum crop gap must be positive when set.")
        if not 0.0 <= self.min_avg_conf <= 1.0 or not 0.0 <= self.crop_pad <= 0.25:
            raise VideoSafetyError("Invalid confidence threshold or crop padding.")
        if not 0.0 < self.min_aspect <= self.max_aspect:
            raise VideoSafetyError("Invalid person-box aspect-ratio range.")
        timed_values = (self.max_wall_time_s, self.max_duration_s, self.default_fps, self.stationary_motion_frac)
        if not all(np.isfinite(value) for value in timed_values) or self.max_wall_time_s <= 0 or self.max_duration_s <= 0 or self.default_fps <= 0:
            raise VideoSafetyError("Time and fallback-FPS limits must be positive.")
        if not 0.0 <= self.stationary_motion_frac <= 1.0:
            raise VideoSafetyError("Stationary-motion fraction must be between 0 and 1.")


def _occlusion(boxes: np.ndarray) -> np.ndarray:
    """Maximum overlap for each box, relative to its own area."""
    if len(boxes) < 2:
        return np.zeros(len(boxes), dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    overlap = intersection / area[:, None]
    np.fill_diagonal(overlap, 0.0)
    return overlap.max(axis=1).astype(np.float32)


def _sharpness(crop: np.ndarray) -> float:
    small = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
    variance = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
    return max(0.05, 1.0 - float(np.exp(-variance / 60.0)))


class _TrackState:
    __slots__ = ("detections", "confs", "edge", "occlusion", "crops", "crop_scores", "crop_confs", "crop_frames", "bag_samples")

    def __init__(self) -> None:
        self.detections: list[tuple[int, float, float, float, float]] = []
        self.confs: list[float] = []
        self.edge: list[bool] = []
        self.occlusion: list[float] = []
        self.crops: list[np.ndarray] = []
        self.crop_scores: list[float] = []
        self.crop_confs: list[float] = []
        self.crop_frames: list[int] = []
        self.bag_samples = 0

    def candidate_slot(self, frame_index: int, gap: int, max_crops: int) -> int:
        if len(self.crops) >= max(1, max_crops // 2):
            distances = np.abs(np.asarray(self.crop_frames) - frame_index)
            nearest = int(distances.argmin())
            if distances[nearest] < gap:
                return nearest
        return -1 if len(self.crops) < max_crops else int(np.argmin(self.crop_scores))

    def offer(self, slot: int, crop: np.ndarray, score: float, confidence: float, frame_index: int) -> None:
        if slot == -1:
            self.crops.append(crop); self.crop_scores.append(score)
            self.crop_confs.append(confidence); self.crop_frames.append(frame_index)
        elif score > self.crop_scores[slot]:
            self.crops[slot] = crop; self.crop_scores[slot] = score
            self.crop_confs[slot] = confidence; self.crop_frames[slot] = frame_index


def _resolve_device(requested: str | None) -> str:
    if requested:
        return str(requested)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _validate_capture(path: Path, cap: cv2.VideoCapture, cfg: AnalyzeConfig) -> tuple[int, int, int, float, float, bool]:
    try:
        byte_size = path.stat().st_size
    except OSError as exc:
        raise VideoSafetyError("The uploaded video could not be inspected.") from exc
    if byte_size < 1 or byte_size > cfg.max_file_bytes:
        raise VideoSafetyError(f"Video must be between 1 byte and {cfg.max_file_bytes // (1024 * 1024)} MB.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if width < 1 or height < 1 or width > cfg.max_width or height > cfg.max_height:
        raise VideoSafetyError(f"Video dimensions must be no larger than {cfg.max_width}×{cfg.max_height}.")
    if total_frames > cfg.max_total_frames:
        raise VideoSafetyError(f"Video has more than {cfg.max_total_frames} frames.")
    fps_estimated = not (reported_fps > 1e-3 and np.isfinite(reported_fps))
    effective_fps = cfg.default_fps if fps_estimated else reported_fps
    if total_frames and total_frames / effective_fps > cfg.max_duration_s:
        raise VideoSafetyError(f"Video duration exceeds {cfg.max_duration_s:.0f} seconds.")
    return width, height, total_frames, reported_fps, effective_fps, fps_estimated


def analyze_video(
    video_path: str | Path,
    *,
    source_name: str,
    source_key: str,
    weights: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    confidence: float = 0.25,
    frame_step: int = 2,
    max_processed_frames: int = 800,
    max_crops: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
    config: AnalyzeConfig | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict], dict]:
    """Return bounded, quality-filtered tracklets from one local video.

    Legacy keyword arguments remain supported. A supplied ``config`` takes
    precedence and activates the complete safety and quality policy.
    """
    cfg = config or AnalyzeConfig(
        weights=weights, tracker=tracker, confidence=confidence,
        frame_step=frame_step, max_processed_frames=max_processed_frames, max_crops=max_crops,
    )
    cfg.validate()
    path = Path(video_path)
    if not path.is_file():
        raise VideoSafetyError("Video file is unavailable.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoSafetyError("Video could not be opened. Use a supported local video format.")

    try:
        width, height, total_frames, reported_fps, fps, fps_estimated = _validate_capture(path, cap, cfg)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install the project dependencies first.") from exc
        device = _resolve_device(cfg.device)
        from .portable_tracker import MotionIoUTracker, resolve_tracker
        actual_tracker, tracker_note = resolve_tracker(cfg.tracker)
        portable = {class_id: MotionIoUTracker() for class_id in cfg.classes} if actual_tracker == "portable-iou" else None
        model = YOLO(cfg.weights)  # Fresh model prevents tracker-state leakage between sources.
        track_kwargs: dict[str, object] = {
            "classes": list(cfg.classes), "conf": cfg.confidence, "max_det": cfg.max_detections_per_frame,
            "imgsz": cfg.imgsz, "verbose": False, "device": device,
        }
        if cfg.iou is not None:
            track_kwargs["iou"] = cfg.iou
        if cfg.half and device.startswith("cuda"):
            track_kwargs["half"] = True

        # Span the full recording while respecting the configured frame budget.
        # A fixed step of 2 used to process only the opening minute of longer
        # clips before declaring the remaining evidence incomplete.
        step = max(cfg.frame_step, (total_frames + cfg.max_processed_frames - 1) // cfg.max_processed_frames) if total_frames else cfg.frame_step
        crop_gap = cfg.min_crop_gap_frames or step * 8
        tracks: dict[tuple[int, int], _TrackState] = {}
        frame_index = -1
        processed = detections_used = skipped_detections = 0
        stop_reason = "end_of_video"
        started = time.perf_counter()

        while True:
            if processed >= cfg.max_processed_frames:
                stop_reason = "max_processed_frames"; break
            if time.perf_counter() - started >= cfg.max_wall_time_s:
                stop_reason = "wall_time_limit"; break
            if should_stop is not None and should_stop():
                stop_reason = "cancelled"; break
            if not cap.grab():
                break
            frame_index += 1
            if frame_index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            processed += 1
            if progress_callback and processed % 5 == 0:
                estimate = min(cfg.max_processed_frames, max(1, (total_frames + step - 1) // step)) if total_frames else cfg.max_processed_frames
                progress_callback(processed, estimate)

            result = (model.predict(frame, **track_kwargs) if portable is not None else
                      model.track(frame, persist=True, tracker=actual_tracker, **track_kwargs))[0]
            boxes_obj = result.boxes
            if boxes_obj is None or (portable is None and boxes_obj.id is None):
                if portable is not None:
                    for tracker_instance in portable.values():
                        tracker_instance.update([])
                continue
            boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
            classes = (boxes_obj.cls.int().cpu().numpy() if boxes_obj.cls is not None else np.zeros(len(boxes), dtype=int))
            if portable is not None:
                ids = np.full(len(boxes), -1, dtype=int)
                for class_id, tracker_instance in portable.items():
                    mask = np.flatnonzero(classes == class_id)
                    ids[mask] = tracker_instance.update(boxes[mask])
            else:
                ids = boxes_obj.id.int().cpu().numpy()
            confs = (boxes_obj.conf.cpu().numpy().astype(np.float32)
                     if boxes_obj.conf is not None else np.ones(len(ids), dtype=np.float32))
            if len(ids) > cfg.max_detections_per_frame:
                keep = np.argsort(confs)[::-1][:cfg.max_detections_per_frame]
                skipped_detections += int(len(ids) - len(keep))
                boxes, ids, confs, classes = boxes[keep], ids[keep], confs[keep], classes[keep]
            occlusion = _occlusion(boxes)

            for box, raw_track_id, class_id, det_conf, overlap in zip(boxes, ids, classes, confs, occlusion):
                bx1, by1, bx2, by2 = (float(value) for value in box)
                box_width, box_height = bx2 - bx1, by2 - by1
                if box_width <= 1 or box_height <= 1:
                    continue
                aspect = box_height / box_width
                if class_id == 0 and not cfg.min_aspect <= aspect <= cfg.max_aspect:
                    continue
                track_id = (int(class_id), int(raw_track_id))
                if track_id not in tracks and len(tracks) >= cfg.max_tracks:
                    skipped_detections += 1
                    continue
                pad_x, pad_y = box_width * cfg.crop_pad, box_height * cfg.crop_pad
                x1, x2 = max(0, int(bx1 - pad_x)), min(width, int(bx2 + pad_x))
                y1, y2 = max(0, int(by1 - pad_y)), min(height, int(by2 + pad_y))
                min_height = cfg.min_crop_height if class_id == 0 else max(16, cfg.min_crop_height // 2)
                min_width = cfg.min_crop_width if class_id == 0 else max(12, cfg.min_crop_width // 2)
                if y2 - y1 < min_height or x2 - x1 < min_width:
                    continue

                state = tracks.setdefault(track_id, _TrackState())
                if class_id == 0:
                    for bag_box in boxes[np.isin(classes, [24, 26, 28])]:
                        center_x, center_y = (bag_box[0] + bag_box[2]) / 2, (bag_box[1] + bag_box[3]) / 2
                        if bx1 - box_width * .3 <= center_x <= bx2 + box_width * .3 and by1 <= center_y <= by2 + box_height * .2:
                            state.bag_samples += 1
                            break
                touches_edge = bx1 <= 2 or by1 <= 2 or bx2 >= width - 2 or by2 >= height - 2
                state.detections.append((frame_index, (bx1 + bx2) / 2, (by1 + by2) / 2, box_width, box_height))
                state.confs.append(float(det_conf)); state.edge.append(bool(touches_edge)); state.occlusion.append(float(overlap))
                detections_used += 1
                area_term = float(np.sqrt(min(1.0, ((y2 - y1) * (x2 - x1)) / (128 * 64))))
                border_term = 0.6 if touches_edge else 1.0
                occlusion_term = 1.0 - 0.6 * min(1.0, float(overlap) / 0.4)
                upper_bound = float(det_conf) * area_term * border_term * occlusion_term
                slot = state.candidate_slot(frame_index, crop_gap, cfg.max_crops)
                if slot != -1 and upper_bound <= state.crop_scores[slot]:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                state.offer(slot, crop, upper_bound * _sharpness(crop), float(det_conf), frame_index)

        has_more_frames = stop_reason == "max_processed_frames" and bool(cap.grab())
        if stop_reason == "max_processed_frames" and not has_more_frames:
            stop_reason = "end_of_video"
    finally:
        cap.release()

    diagonal = float(np.hypot(width, height)) or 1.0
    dropped: defaultdict[str, int] = defaultdict(int)
    output: list[dict] = []
    for (class_id, track_id), state in tracks.items():
        if len(state.detections) < max(2, cfg.min_track_frames):
            dropped["too_short"] += 1; continue
        if not state.crops:
            dropped["no_usable_crops"] += 1; continue
        avg_conf = float(np.mean(state.confs))
        if avg_conf < cfg.min_avg_conf:
            dropped["low_confidence"] += 1; continue
        detections = np.asarray(state.detections, dtype=np.float32)
        frames = detections[:, 0].astype(int)
        first_frame, last_frame = int(frames.min()), int(frames.max())
        centers = detections[:, 1:3]
        path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) if len(centers) > 1 else 0.0
        duration = (last_frame - first_frame) / fps
        stationary = bool(path_length / diagonal < cfg.stationary_motion_frac)
        if stationary and cfg.drop_stationary:
            dropped["stationary"] += 1; continue
        order = np.argsort(state.crop_frames)
        crops = [state.crops[int(index)] for index in order]
        crop_scores = [float(state.crop_scores[int(index)]) for index in order]
        crop_confs = [float(state.crop_confs[int(index)]) for index in order]
        crop_frames = [int(state.crop_frames[int(index)]) for index in order]
        preview_index = int(np.argmax(crop_scores))
        trajectory_indices = (np.linspace(0, len(detections) - 1, cfg.max_trajectory_points).astype(int)
                              if len(detections) > cfg.max_trajectory_points else np.arange(len(detections)))
        output.append({
            "track_id": f"{source_key}_C{class_id}_T{track_id:04d}", "raw_track_id": track_id,
            "class_id": class_id, "object_class": "person" if class_id == 0 else "bag" if class_id in {24, 26, 28} else "vehicle",
            "bag_observation_samples": state.bag_samples if class_id == 0 else 0,
            "source_key": source_key, "source_name": source_name, "camera_id": source_key,
            "frame_count": len(detections), "first_frame": first_frame, "last_frame": last_frame,
            "start_time": first_frame / fps, "end_time": last_frame / fps, "duration_s": float(duration),
            "avg_conf": avg_conf, "crops_bgr": crops, "det_confs": crop_confs,
            "crop_frames": crop_frames, "crop_scores": crop_scores,
            "preview_frame": crop_frames[preview_index],
            "preview_rgb": cv2.cvtColor(crops[preview_index], cv2.COLOR_BGR2RGB),
            "trajectory": detections[trajectory_indices],
            "motion": {
                "path_length_px": path_length,
                "displacement_px": float(np.linalg.norm(centers[-1] - centers[0])),
                "speed_px_per_s": float(path_length / duration) if duration > 0 else 0.0,
                "speed_diagonals_per_s": float(path_length / diagonal / duration) if duration > 0 else 0.0,
                "direction_degrees": float(np.degrees(np.arctan2(centers[-1, 1] - centers[0, 1], centers[-1, 0] - centers[0, 0]))),
                "stationary": stationary,
                "entry_xy_norm": (float(centers[0, 0] / width), float(centers[0, 1] / height)),
                "exit_xy_norm": (float(centers[-1, 0] / width), float(centers[-1, 1] / height)),
            },
            "mean_box_area_frac": float((detections[:, 3] * detections[:, 4]).mean() / (width * height)),
            "coverage": float(min(1.0, len(detections) / ((last_frame - first_frame) / step + 1))),
            "edge_ratio": float(np.mean(state.edge)),
            "occlusion_ratio": float(np.mean(np.asarray(state.occlusion) > 0.3)),
            "embedding": None,
        })

    output.sort(key=lambda item: item["track_id"])
    elapsed = time.perf_counter() - started
    return output, {
        "source_name": source_name, "reported_fps": reported_fps, "fps": fps, "fps_estimated": fps_estimated,
        "total_frames": total_frames, "processed_frames": processed, "frame_size": (width, height),
        "tracklet_count": len(output), "raw_tracks": len(tracks), "dropped_tracks": dict(dropped),
        "detections_used": detections_used, "detections_skipped": skipped_detections,
        "truncated": bool(has_more_frames or stop_reason == "wall_time_limit"), "incomplete": stop_reason != "end_of_video",
        "stop_reason": stop_reason, "device": device, "weights": cfg.weights,
        "effective_frame_step": step,
        "requested_tracker": cfg.tracker, "actual_tracker": actual_tracker, "tracker_note": tracker_note,
        "elapsed_s": float(elapsed), "processing_fps": float(processed / elapsed) if elapsed else 0.0,
    }
