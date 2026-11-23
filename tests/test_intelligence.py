import dataclasses
import numpy as np
import pytest
from fedsight.core import TimeAnchor, RejectionPolicy
from fedsight.cameras import route_context
from fedsight.quality import crop_components
from fedsight.embeddings import _select_diverse, build_osnet_embeddings, OSNetConfig
from fedsight.pair_features import pair_features, rejection
from fedsight.rules import evaluate_rules


def sources(offset=30, uncertainty=0):
    return {"a": {"camera_id": "A", "time_anchor": dataclasses.asdict(TimeAnchor("2026-01-01T00:00:00Z", "UTC", 0, uncertainty, 0, "test"))},
            "b": {"camera_id": "B", "time_anchor": dataclasses.asdict(TimeAnchor("2026-01-01T00:00:00Z", "UTC", offset, uncertainty, 0, "test"))}}


A = {"source_id": "a", "start_time": 0, "end_time": 10}
B = {"source_id": "b", "start_time": 0, "end_time": 10}
EDGES = [{"from_camera": "A", "to_camera": "B", "min_s": 15, "max_s": 25, "provenance": "measured", "confidence": .8}]


def test_route_plausible_unknown_and_contradictory():
    assert route_context(A, B, sources(), EDGES)["status"] == "plausible"
    assert route_context(A, B, sources(offset=100), EDGES)["status"] == "implausible"
    assert route_context(A, B, sources(), [])["status"] == "unknown"
    assert route_context(A, B, sources(uncertainty=15), EDGES)["status"] == "unknown"
    unanchored = sources()
    unanchored["b"]["time_anchor"] = {}
    assert route_context(A, B, unanchored, EDGES)["gap_s"] is None


def test_reversed_sighting_order_uses_correct_directed_edge():
    reverse = [{**EDGES[0], "from_camera": "B", "to_camera": "A"}]
    assert route_context(A, B, sources(offset=-30), reverse)["status"] == "plausible"


def test_time_anchor_timezone_offset_and_drift():
    with pytest.raises(ValueError):
        TimeAnchor("2026-01-01T00:00:00", method="test")
    t = TimeAnchor("2026-01-01T05:30:00+05:30", "Asia/Calcutta", 3, 2, 1000, "test")
    assert t.interval(100)[1] - t.interval(100)[0] == 4
    assert abs(t.interval(100)[0] - 1767225701.1) < .01


def test_quality_components_and_crop_selection():
    rng = np.random.default_rng(1)
    sharp = rng.integers(30, 220, (128, 64, 3), dtype=np.uint8)
    flat = np.full((128, 64, 3), 128, dtype=np.uint8)
    assert crop_components(sharp)["score"] > crop_components(flat)["score"]
    assert crop_components(sharp)["score"] > crop_components(sharp // 20)["score"]
    assert not crop_components(None)["valid"]
    assert _select_diverse([1, 3, 4, 6], np.array([0, .2, .99, .8, .1, .99, .9]), 2) == [3, 6]


def test_open_set_never_calls_cosine_a_probability():
    pair = {"track_a": A, "track_b": B, "similarity": .2}
    features = pair_features(pair, sources(), EDGES)
    assert rejection(features, RejectionPolicy())["state"] == "no_reliable_candidate"
    pair["similarity"] = .9
    features = pair_features(pair, sources(), EDGES)
    assert rejection(features, RejectionPolicy())["state"] == "uncertain"
    assert not features["aice"]["calibrated"]
    with pytest.raises(ValueError):
        pair_features({**pair, "track_b": A}, sources(), EDGES)


def test_embedding_dependency_injection_and_finite_validation():
    rng = np.random.default_rng(1)
    track = {"track_id": "a", "crops_bgr": [rng.integers(0, 255, (128, 64, 3), dtype=np.uint8)]}
    result, note = build_osnet_embeddings([track], OSNetConfig(flip_tta=False), extractor=lambda images: np.ones((len(images), 8)))
    assert np.isclose(np.linalg.norm(result[0]["embedding"]), 1)
    with pytest.raises(ValueError):
        build_osnet_embeddings([{"track_id": "b", "crops_bgr": [np.ones((128,64,3), dtype=np.uint8)]}], OSNetConfig(flip_tta=False), extractor=lambda images: np.full((len(images), 8), np.nan))


def test_batch_rules_line_zone_dwell():
    track = {"track_id": "t", "source_id": "s", "start_time": 0, "end_time": 60, "duration_s": 60,
             "motion": {"stationary": True}, "trajectory": [[0, 10, 30], [1, 80, 30]]}
    rules = [{"id": "r1", "type": "line", "axis": "x", "position": .5},
             {"id": "r2", "type": "zone", "box": [.5, .1, .9, .9]}, {"id": "r3", "type": "dwell", "seconds": 30}]
    assert len(evaluate_rules(track, rules, [100, 100])) == 3


def test_speed_and_wrong_direction_are_image_plane_flags():
    track = {"track_id": "t", "source_id": "s", "start_time": 0, "end_time": 2,
             "motion": {"speed_px_per_s": 25}, "trajectory": [[0, 10, 30], [2, 80, 30]]}
    rules = [{"id": "speed", "type": "speed", "pixels_per_second": 20, "severity": "high"},
             {"id": "direction", "type": "wrong_direction", "axis": "x", "forbidden_direction": "positive"},
             {"id": "opposite", "type": "wrong_direction", "axis": "x", "forbidden_direction": "negative"}]
    events = evaluate_rules(track, rules, [100, 100])
    assert {event["rule_id"] for event in events} == {"speed", "direction"}
    assert next(event for event in events if event["rule_id"] == "speed")["severity"] == "high"
