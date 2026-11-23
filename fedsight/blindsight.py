"""Video-only blind-spot hypotheses with explicit unknowns and provenance."""
from __future__ import annotations

import math
from .core import TimeAnchor
from .cameras import route_context


REGION_KINDS = {"visible", "blind", "obstacle", "occlusion"}


def validate_region(region: dict) -> None:
    if region.get("kind") not in REGION_KINDS:
        raise ValueError("Coverage region must be visible, blind, obstacle or occlusion.")
    if not region.get("label", "").strip() or not region.get("provenance", "").strip():
        raise ValueError("Coverage region needs a label and provenance.")
    box = region.get("box", [])
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("Coverage rectangle needs four finite normalized coordinates.")
    if not box[0] < box[2] or not box[1] < box[3]:
        raise ValueError("Coverage rectangle must have positive width and height.")
    if region["kind"] == "visible" and not region.get("camera_id"):
        raise ValueError("Visible coverage must identify a camera.")
    if region["kind"] == "blind":
        distance = region.get("distance_m")
        if distance is not None and (not math.isfinite(distance) or distance <= 0 or distance > 10000):
            raise ValueError("Measured blind-zone path distance must be positive and at most 10 km.")


def validate_portal(portal: dict) -> None:
    if not portal.get("camera_id") or not portal.get("blind_zone_id"):
        raise ValueError("A portal needs a camera and blind zone.")
    if portal.get("role") not in {"entry", "exit", "both"}:
        raise ValueError("Portal role must be entry, exit or both.")
    validate_region({"kind": "occlusion", "label": portal.get("label", "portal"),
                     "provenance": portal.get("provenance", ""), "box": portal.get("box")})


def _inside(point, box) -> bool:
    return bool(point and len(point) == 2 and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def _ordered_tracks(left: dict, right: dict, sources: dict):
    a, b = sources[left["source_id"]], sources[right["source_id"]]
    ae, bs = TimeAnchor(**a.get("time_anchor", {})).interval(left["end_time"]), TimeAnchor(**b.get("time_anchor", {})).interval(right["start_time"])
    be, az = TimeAnchor(**b.get("time_anchor", {})).interval(right["end_time"]), TimeAnchor(**a.get("time_anchor", {})).interval(left["start_time"])
    if ae and bs and ae[1] <= bs[0]:
        return left, right, ae, bs
    if be and az and be[1] <= az[0]:
        return right, left, be, az
    return None


def _boundary_observation(track: dict, *, leaving: bool) -> dict:
    point = track.get("motion", {}).get("exit_xy_norm" if leaving else "entry_xy_norm")
    near_edge = None if point is None else min(point[0], 1-point[0], point[1], 1-point[1]) <= .1
    return {"track_id": track["id"], "source_id": track["source_id"], "camera_id": track.get("camera_id"),
            "clip_time_s": track["end_time" if leaving else "start_time"],
            "frame": track.get("last_frame" if leaving else "first_frame"),
            "image_xy_norm": point, "near_image_edge": near_edge,
            "meaning": ("Last" if leaving else "First") + " sampled sighting; disappearance/appearance mechanism is unknown."}


def infer_hypothesis(proposal: dict, tracks: dict, sources: dict, transitions: list[dict],
                     regions: list[dict], portals: list[dict], reviews: dict,
                     *, min_speed_mps=.5, max_speed_mps=2.0, competing_proposals=()) -> dict:
    """Return a candidate missing interval, never an asserted location or identity."""
    if not 0 < min_speed_mps <= max_speed_mps <= 4:
        raise ValueError("Use ordered walking-speed bounds between 0 and 4 m/s.")
    left, right = tracks[proposal["left_id"]], tracks[proposal["right_id"]]
    if left.get("object_class", "person") != "person" or right.get("object_class", "person") != "person":
        raise ValueError("BlindSight currently evaluates person tracklets only.")
    route = route_context(left, right, sources, transitions)
    order = _ordered_tracks(left, right, sources)
    support = []
    conflicts = []
    if reviews.get(proposal["id"], {}).get("decision") == "reject":
        conflicts.append("Investigator rejected the appearance link.")
    if competing_proposals:
        conflicts.append(f"{len(competing_proposals)} alternative appearance proposals share a tracklet; compare those candidates.")
    similarity = proposal.get("features", {}).get("values", {}).get("similarity")
    if similarity is not None:
        support.append(f"Visual proposal cosine {similarity:.3f}; uncalibrated and not identity proof.")
    if not order:
        conflicts.append("No defensible ordering: clocks are unaligned, uncertain, or sightings overlap.")
        return {"proposal_id": proposal["id"], "left_id": left["id"], "right_id": right["id"],
                "from_track_id": None, "to_track_id": None, "gap_utc": None, "gap_duration_s": None,
                "disappearance": None, "reappearance": None,
                "route": route, "zones": [], "supporting_evidence": support, "contradictory_evidence": conflicts,
                "assessment": "time_unknown", "status": "requires_review", "meaning": "No unseen presence established."}
    first, second, end, start = order
    gap_bounds = [start[0] - end[1], start[1] - end[0]]
    gap_utc = {"last_seen_bounds": list(end), "next_seen_bounds": list(start)}
    disappearance = _boundary_observation(first, leaving=True)
    reappearance = _boundary_observation(second, leaving=False)
    support.append(f"Last sampled sighting in {first['source_id']} at clip {first['end_time']:.2f}s; next sampled sighting in {second['source_id']} at clip {second['start_time']:.2f}s.")
    if route["status"] == "plausible":
        support.append("Configured camera path intersects the uncertain time gap.")
    elif route["status"] == "implausible":
        conflicts.append("Configured camera travel times conflict with the observed gap; the map may be incomplete.")
    else:
        conflicts.append("Camera route is unverified or clocks cannot support route timing.")
    blind_zones = {r["id"]: r for r in regions if r["kind"] == "blind"}
    possible_zones = []
    for zone_id, zone in blind_zones.items():
        exits = [p for p in portals if p["blind_zone_id"] == zone_id and p["camera_id"] == first.get("camera_id")
                 and p["role"] in {"exit", "both"} and _inside(first.get("motion", {}).get("exit_xy_norm"), p["box"])]
        entries = [p for p in portals if p["blind_zone_id"] == zone_id and p["camera_id"] == second.get("camera_id")
                   and p["role"] in {"entry", "both"} and _inside(second.get("motion", {}).get("entry_xy_norm"), p["box"])]
        if not exits or not entries:
            continue
        distance = zone.get("distance_m")
        walking = [distance / max_speed_mps, distance / min_speed_mps] if distance else None
        compatible = None if walking is None else gap_bounds[1] >= walking[0] and gap_bounds[0] <= walking[1]
        possible_zones.append({"zone_id": zone_id, "label": zone["label"], "exit_portal_ids": [p["id"] for p in exits],
                               "entry_portal_ids": [p["id"] for p in entries], "configured_distance_m": distance,
                               "walking_time_s": walking, "walking_time_compatible": compatible,
                               "provenance": zone["provenance"]})
        if compatible is False:
            conflicts.append(f"Configured distance and walking-speed bounds conflict for {zone['label']}.")
        else:
            support.append(f"Exit and entry points meet configured portals for {zone['label']}.")
    if not possible_zones:
        conflicts.append("No mapped blind zone has matching exit and entry portals for these tracklets.")
    plausible = (reviews.get(proposal["id"], {}).get("decision") != "reject" and route["status"] == "plausible"
                 and any(z["walking_time_compatible"] is not False for z in possible_zones))
    return {"proposal_id": proposal["id"], "left_id": left["id"], "right_id": right["id"],
            "from_track_id": first["id"], "to_track_id": second["id"], "from_camera_id": first.get("camera_id"),
            "to_camera_id": second.get("camera_id"), "gap_utc": gap_utc, "gap_duration_s": gap_bounds,
            "disappearance": disappearance, "reappearance": reappearance,
            "route": route, "zones": possible_zones, "supporting_evidence": support,
            "contradictory_evidence": conflicts, "assessment": "possible_unobserved_movement" if plausible else "unverified_route",
            "status": "requires_review", "meaning": "Hypothesis only; physical presence in the unseen area is not observed."}
