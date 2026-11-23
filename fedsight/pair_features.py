"""Versioned ThreadSight features, transparent AICE and abstention."""
from __future__ import annotations
import math
from .core import FEATURE_VERSION, RejectionPolicy
from .cameras import route_context

FEATURE_NAMES = ["similarity", "fused_similarity", "margin", "quality_min", "consistency_min", "mutual_nn",
                 "time_known", "time_decay", "route_known", "route_plausible"]


def pair_features(pair: dict, sources: dict, transitions: list[dict]) -> dict:
    a, b = pair["track_a"], pair["track_b"]
    if a["source_id"] == b["source_id"]:
        raise ValueError("Candidate pairs must come from different sources.")
    route = route_context(a, b, sources, transitions)
    qa, qb = a.get("embedding_meta", {}), b.get("embedding_meta", {})
    similarity = float(pair["similarity"])
    if not math.isfinite(similarity) or not -1.001 <= similarity <= 1.001:
        raise ValueError("Invalid cosine similarity.")
    time_decay = math.exp(-route["gap_s"] / 300) if route["gap_s"] is not None else None
    # No graph evidence is invented; unknown terms are excluded and weights renormalised.
    components = {"visual": {"value": (max(-1, min(1, similarity)) + 1) / 2, "weight": .7},
                  "time": {"value": time_decay, "weight": .2}, "graph": {"value": None, "weight": .1}}
    available = [v for v in components.values() if v["value"] is not None]
    aice = sum(v["value"] * v["weight"] for v in available) / sum(v["weight"] for v in available)
    values = {"similarity": similarity, "fused_similarity": float(pair.get("fused_similarity", similarity)),
              "margin": float(pair.get("margin", 0)), "quality_min": min(qa.get("mean_quality", 0), qb.get("mean_quality", 0)),
              "consistency_min": min(qa.get("consistency", 0), qb.get("consistency", 0)), "mutual_nn": int(pair.get("mutual_nn", False)),
              "time_known": int(time_decay is not None), "time_decay": time_decay or 0.0,
              "route_known": int(route["status"] != "unknown"), "route_plausible": int(route["status"] == "plausible")}
    return {"version": FEATURE_VERSION, "values": values, "route": route,
            "aice": {"score": aice, "components": components, "tau_s": 300, "calibrated": False,
                     "note": "Unknown terms excluded. Route remains a separate feature; AICE is not added again to classifier inputs."}}


def rejection(features: dict, policy: RejectionPolicy, *, possible_id_switch=False, weights_validated=False) -> dict:
    v, reasons = features["values"], []
    if v["similarity"] < policy.min_similarity:
        return {"state": "no_reliable_candidate", "reasons": ["Visual similarity below the configured review threshold."]}
    if v["quality_min"] < policy.min_quality:
        reasons.append("Low-quality source imagery.")
    if v["margin"] < policy.min_margin:
        reasons.append("Competing candidates have similar scores.")
    if possible_id_switch:
        reasons.append("Within-track embedding inconsistency suggests a possible tracker switch.")
    if features["route"]["status"] != "plausible":
        reasons.append(features["route"]["reason"])
    if not weights_validated:
        reasons.append("Re-ID weights have not been validated on a held-out target dataset.")
    return {"state": "uncertain" if reasons else "review_priority", "reasons": reasons or ["Passes heuristic review gates; human review required."]}
