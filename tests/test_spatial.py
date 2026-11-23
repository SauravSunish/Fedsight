import numpy as np
import pytest

from fedsight.spatial import (
    compute_homography,
    project_point,
    project_feet,
    project_trajectory,
    classify_speed,
    transit_feasibility,
    generate_heatmap,
    aggregate_heatmap,
    make_calibration_record,
    load_homography,
)


def test_homography_exact_square_mapping():
    # Map a 100x100 square to a 500x500 floor plan region
    video_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    floor_pts = [[200, 300], [400, 300], [400, 500], [200, 500]]

    H = compute_homography(video_pts, floor_pts)
    assert H.shape == (3, 3)

    # Check corners map accurately
    for v_pt, f_pt in zip(video_pts, floor_pts):
        fx, fy = project_point(H, v_pt[0], v_pt[1])
        assert pytest.approx(fx, abs=1e-2) == f_pt[0]
        assert pytest.approx(fy, abs=1e-2) == f_pt[1]


def test_homography_degenerate_points_raise():
    # Collinear points should fail
    video_pts = [[0, 0], [10, 10], [20, 20], [30, 30]]
    floor_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    with pytest.raises(ValueError, match="degenerate"):
        compute_homography(video_pts, floor_pts)


def test_project_feet():
    video_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    floor_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    H = compute_homography(video_pts, floor_pts)

    # Bounding box in normalized coords [0.2, 0.2, 0.4, 0.8] on a 100x100 frame
    # foot_x = (20 + 40) / 2 = 30, foot_y = 80
    fx, fy = project_feet(H, [0.2, 0.2, 0.4, 0.8], frame_w=100, frame_h=100)
    assert pytest.approx(fx, abs=1e-3) == 30.0
    assert pytest.approx(fy, abs=1e-3) == 80.0


def test_project_trajectory():
    video_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    floor_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    H = compute_homography(video_pts, floor_pts)

    # Trajectory: [frame, cx, cy, w, h]
    # foot is cx, cy + h/2
    traj = np.array([
        [0, 50, 40, 20, 20],   # foot at (50, 50)
        [25, 60, 40, 20, 20],  # foot at (60, 50)
    ])
    projected = project_trajectory(H, traj, frame_w=100, frame_h=100, fps=25.0)
    assert len(projected) == 2
    assert projected[0]["clip_s"] == 0.0
    assert pytest.approx(projected[0]["floor_x"], abs=1e-3) == 50.0
    assert pytest.approx(projected[0]["floor_y"], abs=1e-3) == 50.0
    assert projected[1]["clip_s"] == 1.0
    assert pytest.approx(projected[1]["floor_x"], abs=1e-3) == 60.0


def test_transit_feasibility_classifications():
    # 10 metres apart, 10 seconds -> 1.0 m/s (likely / normal walk)
    res = transit_feasibility((0, 0), 0.0, (10, 0), 10.0, pixels_per_metre=1.0)
    assert res["speed_mps"] == 1.0
    assert res["feasibility"] in {"very_likely", "likely"}

    # 100 metres apart, 10 seconds -> 10 m/s (implausible sprint)
    res_sprint = transit_feasibility((0, 0), 0.0, (100, 0), 10.0, pixels_per_metre=1.0)
    assert res_sprint["speed_mps"] == 10.0
    assert res_sprint["feasibility"] == "implausible"

    # Simultaneous sighting at different positions
    res_simul = transit_feasibility((0, 0), 100.0, (50, 50), 100.0, pixels_per_metre=1.0)
    assert res_simul["speed_mps"] == float("inf")
    assert res_simul["feasibility"] == "implausible"


def test_heatmap_generation():
    points = [
        {"floor_x": 10, "floor_y": 10},
        {"floor_x": 12, "floor_y": 15},
        {"floor_x": 50, "floor_y": 50},
    ]
    grid = generate_heatmap(points, floor_w=100, floor_h=100, cell_size=20)
    assert grid.shape == (5, 5)
    # Cell (0, 0) should have count 2
    assert grid[0, 0] == 2.0
    # Cell (2, 2) should have count 1
    assert grid[2, 2] == 1.0


def test_calibration_record_serialization():
    video_pts = [[0, 0], [100, 0], [100, 100], [0, 100]]
    floor_pts = [[0, 0], [200, 0], [200, 200], [0, 200]]
    rec = make_calibration_record("case_1", "src_1", "cam_1", video_pts, floor_pts, None, pixels_per_metre=20.0)
    assert rec["case_id"] == "case_1"
    assert rec["pixels_per_metre"] == 20.0

    H = load_homography(rec)
    assert H.shape == (3, 3)
    fx, fy = project_point(H, 50, 50)
    assert pytest.approx(fx, abs=1e-3) == 100.0
    assert pytest.approx(fy, abs=1e-3) == 100.0
