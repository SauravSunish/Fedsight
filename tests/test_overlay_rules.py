import numpy as np
import pytest

from fedsight.rules import evaluate_rules, evaluate_scene_rules, line_cross_direction, point_in_polygon, validate_rule
from fedsight.review_video import annotate_frame
from fedsight.video import AnalyzeConfig


def test_polygon_and_directional_crossing():
    polygon = [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]
    assert point_in_polygon([.5, .5], polygon)
    assert not point_in_polygon([.1, .1], polygon)
    assert line_cross_direction([.1, .5], [.7, .5], "x", .5) == "positive"
    assert line_cross_direction([.7, .5], [.1, .5], "x", .5) == "negative"
    track = {"track_id": "t", "source_id": "s", "start_time": 0, "end_time": 2,
             "first_frame": 0, "last_frame": 20, "trajectory": [[0, 10, 50], [10, 50, 50], [20, 90, 50]]}
    rules = [{"id": "zone", "type": "zone", "polygon": polygon},
             {"id": "line", "type": "line", "axis": "x", "position": .4, "direction": "positive"}]
    events = evaluate_rules(track, rules, [100, 100])
    assert {e["rule_id"] for e in events} == {"zone", "line"}
    assert all(e["frame"] == 10 and e["start_time"] == 1 for e in events)
    with pytest.raises(ValueError):
        validate_rule({"type": "zone", "polygon": [[0, 0], [1, 1], [.5, .5]]})


def test_overlay_highlighting_and_allowlisted_detector_classes():
    image = np.full((250, 320, 3), 100, dtype=np.uint8)
    tracks = [{"id": "a", "object_class": "person", "trajectory": [[0, 100, 100, 40, 80], [10, 110, 100, 40, 80]], "avg_conf": .9},
              {"id": "b", "object_class": "bag", "trajectory": [[0, 150, 150, 30, 30], [10, 155, 150, 30, 30]]}]
    view = annotate_frame(image, tracks, 5, options={"highlight": True, "selected_ids": ["a"], "crowd_threshold": 1},
                          rules=[{"type": "zone", "polygon": [[.1,.1],[.6,.1],[.6,.8],[.1,.8]]}])
    assert view.shape == image.shape and np.any(view != image)
    assert np.all(image == 100)
    AnalyzeConfig(classes=(0, 24, 26, 28)).validate()


def test_unattended_bag_and_crowd_zone_use_sampled_detections():
    bag = {"track_id": "bag", "source_id": "s", "object_class": "bag", "frame_size": [100, 100],
           "first_frame": 0, "last_frame": 30, "start_time": 0, "end_time": 3, "duration_s": 3,
           "motion": {"stationary": True}, "trajectory": [[0, 40, 40], [10, 40, 40], [20, 40, 40], [30, 40, 40]]}
    person = {"track_id": "person", "source_id": "s", "frame_size": [100, 100],
              "first_frame": 0, "last_frame": 30, "start_time": 0, "end_time": 3,
              "trajectory": [[0, 40, 40], [10, 90, 90], [20, 90, 90], [30, 90, 90]]}
    rules = [{"id": "bag-rule", "type": "unattended_object", "seconds": 2, "distance_norm": .1},
             {"id": "crowd", "type": "crowd", "threshold": 1, "zone_box": [0, 0, .5, .5]}]
    events = evaluate_scene_rules([bag, person], rules)
    bag_event = next(e for e in events if e["method"] == "unattended_object")
    assert bag_event["frame"] == 30 and bag_event["start_frame"] == 10
    assert next(e for e in events if e["method"] == "crowd")["frame"] == 0
    bag["motion"]["stationary"] = False
    assert not any(e["method"] == "unattended_object" for e in evaluate_scene_rules([bag, person], rules))
