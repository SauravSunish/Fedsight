"""Bounded directed route reasoning with explicitly uncertain clocks."""
from __future__ import annotations
import math
from .core import TimeAnchor


def validate_transition(edge: dict) -> None:
    if edge["from_camera"] == edge["to_camera"]:
        raise ValueError("Transition endpoints must differ.")
    low, high = edge["min_s"], edge["max_s"]
    if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high:
        raise ValueError("Travel-time bounds must be finite, nonnegative and ordered.")
    if not 0 <= edge.get("confidence", 1) <= 1 or not edge.get("provenance", "").strip():
        raise ValueError("A transition requires confidence and configuration provenance.")


def route_context(left: dict, right: dict, sources: dict, transitions: list[dict], *, max_hops=6, max_paths=200) -> dict:
    a, b = sources[left["source_id"]], sources[right["source_id"]]
    unknown = {"status": "unknown", "gap_s": None, "gap_interval": None, "paths": [], "reason": ""}
    ac, bc = a.get("camera_id"), b.get("camera_id")
    ta, tb = TimeAnchor(**a.get("time_anchor", {})), TimeAnchor(**b.get("time_anchor", {}))
    a_end, b_start = ta.interval(left.get("end_time", 0)), tb.interval(right.get("start_time", 0))
    b_end, a_start = tb.interval(right.get("end_time", 0)), ta.interval(left.get("start_time", 0))
    if a_end is None or b_start is None:
        return {**unknown, "reason": "Recording clocks have no shared, trusted time anchor."}
    # Infer direction only when sighting order is unambiguous after uncertainty.
    if a_end[1] <= b_start[0]:
        first, second, end, start = ac, bc, a_end, b_start
    elif b_end[1] <= a_start[0]:
        first, second, end, start = bc, ac, b_end, a_start
    else:
        return {**unknown, "reason": "Sightings overlap or clock uncertainty prevents ordering."}
    gap = [start[0] - end[1], start[1] - end[0]]
    base = {**unknown, "gap_s": sum(gap) / 2, "gap_interval": gap, "from_camera": first, "to_camera": second}
    if not first or not second or first == second:
        return {**base, "reason": "Missing camera mapping or same-camera sighting."}
    adjacency = {}
    for edge in transitions:
        validate_transition(edge)
        adjacency.setdefault(edge["from_camera"], []).append(edge)
    stack = [(first, [first], 0.0, 0.0, 1.0)]
    paths, expansions, capped = [], 0, False
    while stack:
        camera, path, low, high, confidence = stack.pop()
        expansions += 1
        if expansions > 2000 or len(paths) >= max_paths:
            capped = True
            break
        if camera == second:
            paths.append({"cameras": path, "min_s": low, "max_s": high, "confidence": confidence,
                          "compatible": gap[1] >= low and gap[0] <= high})
            continue
        if len(path) - 1 >= max_hops:
            if adjacency.get(camera):
                capped = True
            continue
        for edge in adjacency.get(camera, []):
            nxt = edge["to_camera"]
            if nxt not in path:
                stack.append((nxt, path + [nxt], low + edge["min_s"], high + edge["max_s"],
                              min(confidence, edge.get("confidence", 1))))
    if any(p["compatible"] for p in paths):
        return {**base, "status": "plausible", "paths": paths, "reason": "A configured route intersects the uncertain travel-time interval."}
    if not paths or capped:
        return {**base, "paths": paths, "reason": "No configured path found; map may be incomplete or search limit reached."}
    return {**base, "status": "implausible", "paths": paths,
            "reason": "Known routes conflict with timing; missing routes may still exist. This is not a hard identity exclusion."}
