import numpy as np
import pytest

from fedsight.video import AnalyzeConfig, VideoSafetyError, _occlusion


def test_default_analysis_configuration_is_valid():
    AnalyzeConfig().validate()
    assert AnalyzeConfig().max_duration_s >= 197.64


@pytest.mark.parametrize(
    "changes",
    [
        {"max_crops": 0},
        {"frame_step": 0},
        {"max_file_bytes": -1},
        {"weights": "../../untrusted.pt"},
        {"tracker": "custom.yaml"},
        {"imgsz": 650},
        {"default_fps": 0},
        {"stationary_motion_frac": 1.1},
    ],
)
def test_invalid_analysis_configuration_is_rejected(changes):
    with pytest.raises(VideoSafetyError):
        AnalyzeConfig(**changes).validate()


def test_occlusion_is_measured_relative_to_each_box():
    boxes = np.asarray([[0, 0, 10, 10], [5, 0, 15, 10], [30, 30, 40, 40]], dtype=np.float32)
    assert np.allclose(_occlusion(boxes), [0.5, 0.5, 0.0])


def test_occlusion_handles_a_single_detection():
    assert np.array_equal(_occlusion(np.asarray([[0, 0, 10, 10]], dtype=np.float32)), np.asarray([0.0], dtype=np.float32))
