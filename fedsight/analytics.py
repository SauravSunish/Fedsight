"""Descriptive statistics on sampled tracklets, kept separate by camera."""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np


def activity_bins(tracks: list[dict], *, bin_seconds: int = 30) -> list[dict]:
    if bin_seconds < 1:
        raise ValueError("Bin width must be positive.")
    bins = defaultdict(set)
    for track in tracks:
        start, end = float(track.get("start_time", 0)), float(track.get("end_time", 0))
        for interval in range(int(start // bin_seconds), int(end // bin_seconds) + 1):
            bins[(track["source_id"], interval)].add(track["track_id"])
    return [{"source_id": source, "start_s": interval * bin_seconds, "visible_tracklets": len(ids)}
            for (source, interval), ids in sorted(bins.items())]


def common_paths(tracks: list[dict], *, grid: int = 4) -> list[dict]:
    if grid < 2:
        raise ValueError("Grid must have at least two cells per axis.")
    paths = Counter()
    for track in tracks:
        motion = track.get("motion", {})
        entry, exit_ = motion.get("entry_xy_norm"), motion.get("exit_xy_norm")
        if entry is None or exit_ is None:
            continue
        cell = lambda xy: (min(grid-1, max(0, int(xy[0]*grid))), min(grid-1, max(0, int(xy[1]*grid))))
        paths[(track["source_id"], cell(entry), cell(exit_))] += 1
    return [{"source_id": source, "entry_cell": f"{a[0]},{a[1]}", "exit_cell": f"{b[0]},{b[1]}", "tracklets": count}
            for (source, a, b), count in paths.most_common()]


def dwell_summary(tracks: list[dict]) -> list[dict]:
    output = []
    for track in tracks:
        if track.get("motion", {}).get("stationary", False):
            output.append({"source_id": track["source_id"], "track_id": track["track_id"],
                           "visible_duration_s": round(float(track.get("duration_s", 0)), 2),
                           "start_s": round(float(track.get("start_time", 0)), 2)})
    return sorted(output, key=lambda row: -row["visible_duration_s"])
