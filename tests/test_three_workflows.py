import numpy as np

from fedsight.analytics import activity_bins, common_paths, dwell_summary
from fedsight.rules import evaluate_scene_rules
from fedsight.vehicle import vehicle_descriptor, describe_vehicles
from fedsight.video import AnalyzeConfig, VideoSafetyError
from fedsight.video import analyze_video
from fedsight.live import LiveSession


def test_vehicle_descriptors_are_finite_and_colour_sensitive():
    red = np.full((64, 96, 3), (0, 0, 255), dtype=np.uint8)
    blue = np.full((64, 96, 3), (255, 0, 0), dtype=np.uint8)
    r, b = vehicle_descriptor(red), vehicle_descriptor(blue)
    assert np.isclose(np.linalg.norm(r), 1)
    assert np.isfinite(r).all()
    assert r @ r > r @ b
    track = {"crops_bgr": [red, red.copy()]}
    describe_vehicles([track])
    assert track["embedding_meta"]["trained_reid"] is False
    assert len(track["embeddings"]) == 2


def test_scene_crowd_counts_distinct_person_tracks_and_ignores_vehicles():
    base = {"source_id": "s", "camera_id": "c", "start_time": 0, "end_time": 1,
            "trajectory": [[4, 10, 10], [5, 12, 10]]}
    tracks = [{**base, "track_id": "p1", "object_class": "person"},
              {**base, "track_id": "p2", "object_class": "person"},
              {**base, "track_id": "v1", "object_class": "vehicle"}]
    rules = [{"id": "crowd", "type": "crowd", "threshold": 2},
             {"id": "high", "type": "count", "threshold": 3}]
    events = evaluate_scene_rules(tracks, rules)
    assert len(events) == 1
    assert events[0]["rule_id"] == "crowd"
    assert events[0]["observed_tracklets"] == 2


def test_descriptive_analytics_keep_sources_and_stationary_tracks_separate():
    tracks = [{"source_id": "a", "track_id": "1", "start_time": 5, "end_time": 35,
               "duration_s": 30, "motion": {"stationary": True, "entry_xy_norm": [.1, .1], "exit_xy_norm": [.8, .8]}},
              {"source_id": "b", "track_id": "2", "start_time": 5, "end_time": 15,
               "duration_s": 10, "motion": {"stationary": False, "entry_xy_norm": [.1, .1], "exit_xy_norm": [.8, .8]}}]
    bins = activity_bins(tracks, bin_seconds=30)
    assert [(b["source_id"], b["start_s"]) for b in bins] == [("a", 0), ("a", 30), ("b", 0)]
    assert len(common_paths(tracks)) == 2
    assert [d["track_id"] for d in dwell_summary(tracks)] == ["1"]


def test_detector_class_allowlist():
    AnalyzeConfig(classes=(0, 2, 7)).validate()
    try:
        AnalyzeConfig(classes=(0, 16)).validate()
    except VideoSafetyError:
        pass
    else:
        raise AssertionError("unsupported detector class should be rejected")


def test_live_local_camera_alerts_on_zone_entry(monkeypatch):
    class Capture:
        released = False
        def isOpened(self): return True
        def read(self): return True, np.zeros((100, 100, 3), dtype=np.uint8)
        def release(self): self.released = True
    capture = Capture()
    monkeypatch.setattr("fedsight.live.cv2.VideoCapture", lambda index: capture)
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data, dtype=float)
        def cpu(self): return self
        def numpy(self): return self.data
    class Model:
        calls = 0
        def predict(self, frame, **kwargs):
            self.calls += 1
            box = [5, 25, 25, 75] if self.calls == 1 else [45, 25, 65, 75]
            return [type("Result", (), {"boxes": type("Boxes", (), {"xyxy": Tensor([box])})()})()]
    live = LiveSession(0)
    rule = {"id": "zone", "type": "zone", "label": "Door", "box": [.4, .2, .8, .8]}
    assert live.step(Model(), [rule])[2] == []
    # Use one model instance so the second detection enters the zone.
    model = Model()
    live.step(model, [rule])
    alerts = live.step(model, [rule])[2]
    assert len(alerts) == 1 and alerts[0]["rule"] == "Door"
    live.close()
    assert capture.released


def test_video_indexes_person_and_vehicle_as_separate_tracks(tmp_path, monkeypatch):
    class Capture:
        frame = 0
        def isOpened(self): return True
        def get(self, prop):
            import cv2
            return {cv2.CAP_PROP_FRAME_WIDTH: 160, cv2.CAP_PROP_FRAME_HEIGHT: 120,
                    cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 10}.get(prop, 0)
        def grab(self):
            self.frame += 1
            return self.frame <= 3
        def retrieve(self): return True, np.full((120, 160, 3), 128, dtype=np.uint8)
        def release(self): pass
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return self
        def numpy(self): return self.data
        def int(self): return Tensor(self.data.astype(int))
    class Model:
        def __init__(self, weights): pass
        def predict(self, frame, **kwargs):
            assert kwargs["classes"] == [0, 2]
            boxes = type("Boxes", (), {"xyxy": Tensor([[20, 10, 60, 110], [75, 30, 145, 95]]),
                                      "cls": Tensor([0, 2]), "conf": Tensor([.9, .8])})()
            return [type("Result", (), {"boxes": boxes})()]
    import ultralytics
    monkeypatch.setattr("fedsight.video.cv2.VideoCapture", lambda path: Capture())
    monkeypatch.setattr(ultralytics, "YOLO", Model)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stub")
    tracks, meta = analyze_video(video, source_name="clip", source_key="s",
                                 config=AnalyzeConfig(classes=(0, 2), tracker="portable-iou", frame_step=1, max_processed_frames=4))
    assert {t["object_class"] for t in tracks} == {"person", "vehicle"}
    assert len({t["track_id"] for t in tracks}) == 2
    assert meta["tracklet_count"] == 2


def test_review_reel_encodes_timestamped_track_preview(tmp_path):
    import cv2
    from fedsight.digest import review_reel
    import pytest
    image = np.full((80, 50, 3), (0, 120, 210), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    service = type("Service", (), {"work": tmp_path, "read_asset": lambda self, asset, actor: encoded})()
    track = {"source_id": "s", "track_id": "t1", "start_time": 2., "end_time": 4.,
             "preview_asset": {"asset_id": "a"}, "object_class": "person"}
    try:
        reel = review_reel(service, [track], {"s": {"name": "Camera A"}}, "tester")
    except OSError as exc:
        pytest.skip(str(exc))
    assert len(reel) > 100
    assert b"ftyp" in reel[:64]
    assert not list(tmp_path.glob("*.mp4"))
