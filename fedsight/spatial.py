"""Spatial intelligence: floor-plan homography, trajectory projection and transit feasibility.

Projects tracklet bounding-box feet positions onto a calibrated floor plan using
a 4-point planar homography.  Provides transit-feasibility scoring, trajectory
rendering and per-cell heatmap aggregation for visual investigation.

This module never infers identity; it produces spatial leads for human review.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .core import TimeAnchor, uid, now


# ---------------------------------------------------------------------------
# Homography calibration
# ---------------------------------------------------------------------------

def compute_homography(video_points: list[list[float]],
                       floor_points: list[list[float]]) -> np.ndarray:
    """Compute a 3×3 perspective transform from ≥4 point correspondences.

    Parameters
    ----------
    video_points : list of [x, y] in pixel coordinates on the video frame.
    floor_points : list of [x, y] in floor-plan coordinates (pixels or metres).

    Returns
    -------
    H : 3×3 float64 ndarray (the homography matrix).

    Raises
    ------
    ValueError
        If fewer than 4 points or the transform is degenerate.
    """
    if len(video_points) < 4 or len(floor_points) < 4:
        raise ValueError("At least 4 calibration point pairs are required.")
    if len(video_points) != len(floor_points):
        raise ValueError("Video and floor-plan point counts must match.")
    src = np.array(video_points, dtype=np.float64)
    dst = np.array(floor_points, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, method=0)  # exact (no RANSAC needed for hand-picked points)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3 or abs(np.linalg.det(H)) < 1e-9:
        raise ValueError("The selected points produce a degenerate transform.  Choose 4 non-collinear points.")
    return H


def project_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Project a single video-frame point (x, y) onto the floor plan."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    result = H @ pt
    if abs(result[2]) < 1e-12:
        raise ValueError("Projection at infinity; check calibration points.")
    return float(result[0] / result[2]), float(result[1] / result[2])


def project_feet(H: np.ndarray, bbox: list[float], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Project the bottom-centre of a bounding box (the 'feet' contact point).

    bbox is [x1, y1, x2, y2] in *normalised* [0,1] coordinates (as stored
    in FedSight trajectory arrays) or absolute pixel coordinates.
    """
    if all(0 <= v <= 1 for v in bbox[:4]):
        x1, y1, x2, y2 = bbox[0] * frame_w, bbox[1] * frame_h, bbox[2] * frame_w, bbox[3] * frame_h
    else:
        x1, y1, x2, y2 = bbox[:4]
    foot_x = (x1 + x2) / 2.0
    foot_y = y2  # bottom edge
    return project_point(H, foot_x, foot_y)


# ---------------------------------------------------------------------------
# Trajectory projection
# ---------------------------------------------------------------------------

def project_trajectory(H: np.ndarray, trajectory: np.ndarray,
                       frame_w: int, frame_h: int, fps: float = 25.0
                       ) -> list[dict[str, float]]:
    """Project a full trajectory array onto floor-plan coordinates.

    Parameters
    ----------
    H : 3×3 homography matrix.
    trajectory : N×5 array  [frame, cx, cy, w, h]  (pixel coordinates).
    frame_w, frame_h : source video dimensions.
    fps : frames per second for time computation.

    Returns
    -------
    List of dicts with keys: clip_s, floor_x, floor_y.
    """
    if trajectory is None or len(trajectory) == 0:
        return []
    points = []
    for row in np.asarray(trajectory, dtype=np.float64):
        frame, cx, cy, w, h = row[:5]
        # Reconstruct bbox bottom-centre from centre + size
        foot_x = cx
        foot_y = cy + h / 2.0
        try:
            fx, fy = project_point(H, foot_x, foot_y)
            points.append({"clip_s": float(frame / fps), "floor_x": float(fx), "floor_y": float(fy)})
        except ValueError:
            continue
    return points


# ---------------------------------------------------------------------------
# Transit feasibility
# ---------------------------------------------------------------------------

# Human locomotion reference speeds (m/s)
WALK_SLOW = 0.8    # elderly / cautious
WALK_NORMAL = 1.4  # average adult
WALK_FAST = 2.0    # brisk walk
JOG = 3.5          # jogging
SPRINT = 8.0       # near-maximum sprint


def classify_speed(speed_mps: float) -> dict[str, Any]:
    """Classify a required transit speed into a human-readable assessment."""
    if speed_mps < 0:
        return {"feasibility": "error", "label": "Invalid speed", "color": "#888888"}
    if speed_mps <= WALK_SLOW:
        return {"feasibility": "very_likely", "label": "Slow walk", "color": "#22c55e"}
    if speed_mps <= WALK_NORMAL:
        return {"feasibility": "likely", "label": "Normal walk", "color": "#4ade80"}
    if speed_mps <= WALK_FAST:
        return {"feasibility": "plausible", "label": "Brisk walk", "color": "#facc15"}
    if speed_mps <= JOG:
        return {"feasibility": "possible", "label": "Jogging required", "color": "#fb923c"}
    if speed_mps <= SPRINT:
        return {"feasibility": "unlikely", "label": "Sprint required", "color": "#ef4444"}
    return {"feasibility": "implausible", "label": "Exceeds human sprint — likely false positive", "color": "#dc2626"}


def transit_feasibility(floor_a: tuple[float, float], time_a: float,
                        floor_b: tuple[float, float], time_b: float,
                        *, pixels_per_metre: float = 1.0) -> dict[str, Any]:
    """Compute transit feasibility between two projected floor-plan positions.

    Parameters
    ----------
    floor_a, floor_b : (x, y) floor-plan coordinates.
    time_a, time_b : UTC timestamps (epoch seconds).
    pixels_per_metre : scale factor to convert floor-plan pixel distance to metres.

    Returns
    -------
    Dict with distance_m, delta_s, speed_mps, and classification.
    """
    dx = floor_b[0] - floor_a[0]
    dy = floor_b[1] - floor_a[1]
    distance_px = math.sqrt(dx * dx + dy * dy)
    distance_m = distance_px / pixels_per_metre if pixels_per_metre > 0 else distance_px
    delta_s = abs(time_b - time_a)
    if delta_s < 0.01:
        return {"distance_m": distance_m, "delta_s": delta_s, "speed_mps": float("inf"),
                **classify_speed(float("inf")), "note": "Simultaneous sightings — same person cannot be in two places."}
    speed_mps = distance_m / delta_s
    result = classify_speed(speed_mps)
    return {"distance_m": round(distance_m, 2), "delta_s": round(delta_s, 2),
            "speed_mps": round(speed_mps, 3), **result}


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap(projected_points: list[dict], floor_w: int, floor_h: int,
                     *, cell_size: int = 20) -> np.ndarray:
    """Build a 2D density grid from projected trajectory points.

    Returns an (rows, cols) float32 array of visit counts per cell, suitable
    for rendering as a semi-transparent overlay.
    """
    cols = max(1, floor_w // cell_size)
    rows = max(1, floor_h // cell_size)
    grid = np.zeros((rows, cols), dtype=np.float32)
    for pt in projected_points:
        c = int(pt["floor_x"] / cell_size)
        r = int(pt["floor_y"] / cell_size)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] += 1
    return grid


# ---------------------------------------------------------------------------
# Calibration record helpers
# ---------------------------------------------------------------------------

def make_calibration_record(case_id: str, source_id: str, camera_id: str | None,
                            video_points: list[list[float]],
                            floor_points: list[list[float]],
                            floor_plan_asset: dict | None,
                            *, pixels_per_metre: float = 1.0,
                            actor: str = "local-reviewer",
                            provenance: str = "") -> dict:
    """Construct a calibration record dict for repo.put('floor_calibration', ...)."""
    H = compute_homography(video_points, floor_points)
    return {
        "id": uid("cal"),
        "case_id": case_id,
        "source_id": source_id,
        "camera_id": camera_id,
        "video_points": [list(map(float, p)) for p in video_points],
        "floor_points": [list(map(float, p)) for p in floor_points],
        "homography": H.tolist(),
        "pixels_per_metre": float(pixels_per_metre),
        "floor_plan_asset": floor_plan_asset,
        "provenance": provenance or "Manual 4-point calibration",
        "actor": actor,
        "created_at": now(),
    }


def load_homography(calibration: dict) -> np.ndarray:
    """Reconstruct the numpy homography matrix from a stored calibration record."""
    return np.array(calibration["homography"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Batch projection for a full case
# ---------------------------------------------------------------------------

def project_case_tracks(calibrations: list[dict], tracks: list[dict],
                        sources: dict, jobs: dict) -> dict[str, list[dict]]:
    """Project every track's trajectory onto the floor plan using stored calibrations.

    Returns a dict mapping track_id → list of {clip_s, floor_x, floor_y}.
    Only tracks whose source has a calibration are projected.
    """
    cal_by_source = {c["source_id"]: c for c in calibrations}
    results = {}
    for track in tracks:
        cal = cal_by_source.get(track["source_id"])
        if cal is None:
            continue
        H = load_homography(cal)
        trajectory = track.get("trajectory")
        if trajectory is None or len(trajectory) == 0:
            continue
        job = jobs.get(track.get("job_id"), {})
        meta = job.get("meta", {})
        fps = float(meta.get("fps") or 25)
        frame_size = track.get("frame_size", meta.get("frame_size", [640, 360]))
        frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
        projected = project_trajectory(H, np.asarray(trajectory), frame_w, frame_h, fps)
        if projected:
            results[track["id"]] = projected
    return results


def aggregate_heatmap(all_projected: dict[str, list[dict]],
                      floor_w: int, floor_h: int,
                      *, cell_size: int = 20) -> np.ndarray:
    """Combine all projected trajectories into a single density heatmap."""
    all_points = []
    for points in all_projected.values():
        all_points.extend(points)
    return generate_heatmap(all_points, floor_w, floor_h, cell_size=cell_size)


# ---------------------------------------------------------------------------
# Cross-camera transit analysis for proposals
# ---------------------------------------------------------------------------

def analyze_proposal_transit(proposal: dict, tracks: dict, sources: dict,
                             calibrations: list[dict], jobs: dict,
                             *, pixels_per_metre: float = 1.0) -> dict[str, Any] | None:
    """Compute transit feasibility for a cross-camera proposal.

    Returns None if either track lacks a calibration or trajectory.
    """
    left = tracks.get(proposal["left_id"])
    right = tracks.get(proposal["right_id"])
    if not left or not right:
        return None

    cal_by_source = {c["source_id"]: c for c in calibrations}
    cal_a = cal_by_source.get(left["source_id"])
    cal_b = cal_by_source.get(right["source_id"])
    if not cal_a or not cal_b:
        return None

    H_a = load_homography(cal_a)
    H_b = load_homography(cal_b)

    # Get frame sizes
    job_a = jobs.get(left.get("job_id"), {})
    job_b = jobs.get(right.get("job_id"), {})
    fs_a = left.get("frame_size", job_a.get("meta", {}).get("frame_size", [640, 360]))
    fs_b = right.get("frame_size", job_b.get("meta", {}).get("frame_size", [640, 360]))

    # Project exit point of first track and entry point of second
    traj_a = left.get("trajectory")
    traj_b = right.get("trajectory")
    if traj_a is None or len(traj_a) == 0 or traj_b is None or len(traj_b) == 0:
        return None

    traj_a = np.asarray(traj_a, dtype=np.float64)
    traj_b = np.asarray(traj_b, dtype=np.float64)

    # Last position of track A, first position of track B
    last_a = traj_a[-1]
    first_b = traj_b[0]

    try:
        exit_floor = project_point(H_a, last_a[1], last_a[2] + last_a[4] / 2)  # cx, cy+h/2
        entry_floor = project_point(H_b, first_b[1], first_b[2] + first_b[4] / 2)
    except ValueError:
        return None

    # Compute UTC times
    src_a = sources.get(left["source_id"], {})
    src_b = sources.get(right["source_id"], {})
    anchor_a = TimeAnchor(**src_a.get("time_anchor", {}))
    anchor_b = TimeAnchor(**src_b.get("time_anchor", {}))

    time_a = anchor_a.interval(left["end_time"])
    time_b = anchor_b.interval(right["start_time"])

    if time_a is None or time_b is None:
        return None

    # Use centre of uncertainty intervals
    utc_a = (time_a[0] + time_a[1]) / 2
    utc_b = (time_b[0] + time_b[1]) / 2

    ppm = cal_a.get("pixels_per_metre", pixels_per_metre)

    result = transit_feasibility(exit_floor, utc_a, entry_floor, utc_b, pixels_per_metre=ppm)
    result["exit_floor"] = exit_floor
    result["entry_floor"] = entry_floor
    result["proposal_id"] = proposal["id"]
    result["left_id"] = proposal["left_id"]
    result["right_id"] = proposal["right_id"]
    return result
