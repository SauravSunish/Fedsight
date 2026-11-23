import numpy as np

from fedsight.review_video import annotate_frame, box_at_frame, track_density, similar_tracks, render_annotated_clip


def test_sampled_boxes_interpolate_only_inside_supported_gap():
    track = {"id": "track_1", "trajectory": [[10, 30, 40, 20, 40], [20, 50, 40, 20, 40]]}
    assert box_at_frame(track, 10) == (20, 20, 40, 60)
    assert box_at_frame(track, 15) == (30, 20, 50, 60)
    assert box_at_frame(track, 9) is None
    assert box_at_frame(track, 15, max_gap=5) is None
    original = np.zeros((80, 100, 3), dtype=np.uint8)
    result = annotate_frame(original, [track], 15)
    assert result.any() and not original.any()


def test_density_counts_tracklets_in_overlapping_bins():
    tracks = [{"start_time": 1, "end_time": 5}, {"start_time": 4, "end_time": 9}]
    assert [item["tracklets"] for item in track_density(tracks, 10, bins=2)] == [2, 2]


def test_similarity_skips_mismatched_categories_and_invalid_embeddings():
    reference = {"id": "a", "embedding": [1, 0], "object_class": "person"}
    candidates = [reference, {"id": "b", "embedding": [.8, .2], "object_class": "person"},
                  {"id": "c", "embedding": [1, 0], "object_class": "vehicle"},
                  {"id": "d", "embedding": [0, 0], "object_class": "person"}]
    matches = similar_tracks(reference, candidates)
    assert [track["id"] for track, score in matches] == ["b"]
    assert .9 < matches[0][1] < 1


def test_annotated_clip_preserves_source_frame_timing(tmp_path):
    import cv2
    import pytest
    source = str(tmp_path / "source.avi")
    output = str(tmp_path / "annotated.mp4")
    writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for _ in range(30):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()
    track = {"id": "track_1", "trajectory": [[10, 30, 40, 20, 40], [20, 50, 40, 20, 40]]}
    info = render_annotated_clip(source, output, [track], start_s=1, duration_s=1)
    assert info["start_frame"] == 10 and info["frames"] == 10
    cap = cv2.VideoCapture(output)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.any()
    with pytest.raises(ValueError):
        render_annotated_clip(source, output, [track], start_s=0, duration_s=301)


def test_annotated_clip_can_exceed_previous_duration_and_frame_limits(tmp_path):
    import cv2
    import pytest
    source, output = str(tmp_path / "long.avi"), str(tmp_path / "long.mp4")
    writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 120, (32, 32))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for _ in range(3720):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    info = render_annotated_clip(source, output, [], start_s=0, duration_s=31,
                                 options={"hud": False, "timestamp": False})
    assert info["frames"] == 3720
    assert info["duration_s"] == 31


def test_background_render_preserves_audited_derived_asset(tmp_path, monkeypatch):
    import time
    from pathlib import Path
    from fedsight.render_queue import RenderQueue
    from fedsight.service import FedSight
    service = FedSight(tmp_path)
    case_id = service.repo.create_case("test", "", "reviewer")["id"]
    source = service.ingest(case_id, "clip.mp4", b"original", "reviewer")
    def fake_renderer(input_path, output_path, tracks, **kwargs):
        assert Path(input_path).read_bytes() == b"original"
        Path(output_path).write_bytes(b"annotated output")
        return {"codec": "h264"}
    monkeypatch.setattr("fedsight.render_queue.render_annotated_clip", fake_renderer)
    queue = RenderQueue()
    task = queue.start(service, case_id, source, "reviewer", [], start_s=0, duration_s=2,
                       options={}, time_anchor={}, validated_ids=set(), rules=[])
    deadline = time.monotonic() + 5
    while task.snapshot()["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(.01)
    state = task.snapshot()
    assert state["status"] == "completed"
    assert service.read_asset(state["result"]["asset"]) == b"annotated output"
    assert any(a["role"] == "annotated_clip" and a["parent_id"] == source["asset"]["asset_id"]
               for a in service.repo.list(case_id, "asset"))
    assert service.repo.verify()["ok"]
    assert list(service.work.iterdir()) == []


def test_offline_demo_video_is_generated_without_overwriting(tmp_path):
    import cv2
    import pytest
    from scripts.create_sample_video import create_sample
    output = tmp_path / "demo.mp4"
    create_sample(output)
    cap = cv2.VideoCapture(str(output))
    ok, frame = cap.read()
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    assert ok and frame is not None and count == 200
    with pytest.raises(FileExistsError):
        create_sample(output)
