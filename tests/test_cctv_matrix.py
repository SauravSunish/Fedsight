"""Unit tests for the synchronized Multi-Camera CCTV Video Matrix."""
import pytest
import numpy as np
from PIL import Image

from fedsight.cctv_matrix import (
    get_active_tracks_at_time,
    interpolate_track_box,
    draw_cctv_frame_overlays,
    generate_synthetic_cctv_frame,
    compose_cctv_matrix,
    extract_or_render_camera_frame,
)
from fedsight.core import TimeAnchor


def test_missing_original_never_appears_as_demo_footage():
    with pytest.raises(ValueError, match="Original video unavailable"):
        extract_or_render_camera_frame(None, {}, "Camera A", [], 0)


def test_get_active_tracks_at_time():
    tracks = [
        {"id": "t1", "source_id": "s1", "start_time": 2.0, "end_time": 10.0},
        {"id": "t2", "source_id": "s1", "start_time": 12.0, "end_time": 20.0},
        {"id": "t3", "source_id": "s2", "start_time": 2.0, "end_time": 10.0},
    ]

    active_s1_at_5 = get_active_tracks_at_time(tracks, "s1", 5.0)
    assert len(active_s1_at_5) == 1
    assert active_s1_at_5[0]["id"] == "t1"

    active_s1_at_15 = get_active_tracks_at_time(tracks, "s1", 15.0)
    assert len(active_s1_at_15) == 1
    assert active_s1_at_15[0]["id"] == "t2"

    active_s1_at_25 = get_active_tracks_at_time(tracks, "s1", 25.0)
    assert len(active_s1_at_25) == 0


def test_interpolate_track_box():
    # Trajectory with 3 points
    traj = np.array([
        [0, 100, 150, 40, 80],
        [25, 200, 150, 40, 80],
        [50, 300, 150, 40, 80],
    ])
    track = {
        "id": "trk_01",
        "start_time": 0.0,
        "end_time": 2.0,
        "trajectory": traj,
    }

    # At t = 1.0s (frame 25)
    x, y, w, h, speed = interpolate_track_box(track, 1.0, fps=25.0)
    assert x == 200
    assert y == 150
    assert w == 40
    assert h == 80
    assert speed > 0


def test_generate_synthetic_cctv_frame():
    track = {
        "id": "trk_demo",
        "start_time": 0.0,
        "end_time": 10.0,
        "attributes": {"color": "blue"},
        "trajectory": np.array([[0, 200, 180, 50, 100], [250, 400, 180, 50, 100]]),
    }
    anchor = TimeAnchor("2026-01-01T10:00:00+00:00", "UTC", 0, 1, 0, "test")

    # Frame with active target
    frame_with_tgt = generate_synthetic_cctv_frame("CAM 01 // NORTH CORRIDOR", 2.0, [track],
                                                   time_anchor=anchor, fps=25.0, width=480, height=270)
    assert isinstance(frame_with_tgt, Image.Image)
    assert frame_with_tgt.size == (480, 270)

    # Frame without active target
    frame_no_tgt = generate_synthetic_cctv_frame("CAM 02 // EAST CORRIDOR", 2.0, [],
                                                 time_anchor=anchor, fps=25.0, width=480, height=270)
    assert isinstance(frame_no_tgt, Image.Image)
    assert frame_no_tgt.size == (480, 270)


def test_compose_cctv_matrix():
    f1 = Image.new("RGB", (320, 180), (10, 20, 30))
    f2 = Image.new("RGB", (320, 180), (20, 30, 40))
    f3 = Image.new("RGB", (320, 180), (30, 40, 50))
    f4 = Image.new("RGB", (320, 180), (40, 50, 60))

    matrix = compose_cctv_matrix([("c1", f1), ("c2", f2), ("c3", f3), ("c4", f4)], grid=(2, 2))
    assert isinstance(matrix, Image.Image)
    # 2 cols * 320 + 6 gutter = 646, 2 rows * 180 + 6 = 366
    assert matrix.size == (646, 366)
