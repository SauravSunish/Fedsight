import io

import numpy as np
import pytest
from PIL import Image

from fedsight.semantic_search import index_case, search_case


class Repo:
    def __init__(self): self.rows = []
    def list(self, case, kind): return [r for r in self.rows if r["case_id"] == case]
    def put(self, kind, record, actor):
        self.rows.append(record)
        return record


class Model:
    model_id = "test-model"
    def image_vectors(self, images):
        return np.array([[1, 0] if image.getpixel((0, 0))[0] > 100 else [0, 1] for image in images])
    def text_vector(self, text): return np.array([1, 0])


class Service:
    def __init__(self):
        self.repo = Repo()
        self.tracks = {
            "a": [{"id": "a1", "source_id": "s1", "camera_id": "cam1", "start_time": 1, "end_time": 2,
                   "preview_asset": {"sha256": "red"}, "trajectory": [[0, 10, 10, 15, 40]]},
                  {"id": "a2", "source_id": "s2", "camera_id": "cam2", "start_time": 5, "end_time": 6,
                   "preview_asset": {"sha256": "blue"}, "trajectory": [[0, 10, 10, 20, 90]]}],
            "b": [{"id": "b1", "source_id": "s3", "camera_id": "cam1", "start_time": 1, "end_time": 2,
                   "preview_asset": {"sha256": "red"}}],
        }
    def active_tracks(self, case): return self.tracks[case]
    def blind_alignments(self, case): return {}
    def read_asset(self, asset, actor):
        buffer = io.BytesIO()
        Image.new("RGB", (5, 5), "red" if asset["sha256"] == "red" else "blue").save(buffer, format="PNG")
        return buffer.getvalue()


def test_index_is_incremental_and_search_is_case_scoped():
    service, model = Service(), Model()
    assert index_case(service, "a", "reviewer", model)["indexed"] == 2
    assert index_case(service, "a", "reviewer", model)["indexed"] == 0
    index_case(service, "b", "reviewer", model)
    ranked = search_case(service, "a", model, "red jacket")
    assert [r["track"]["id"] for r in ranked["results"]] == ["a1", "a2"]
    assert ranked["results"][0]["score"] > ranked["results"][1]["score"]
    assert [r["track"]["id"] for r in search_case(service, "a", model, "red", camera_id="cam2")["results"]] == ["a2"]
    assert [r["track"]["id"] for r in search_case(service, "a", model, "red", min_height=50)["results"]] == ["a2"]
    assert [r["track"]["id"] for r in search_case(service, "a", model, "red", start_s=4)["results"]] == ["a2"]
    service.tracks["a"] = [service.tracks["a"][1]]
    assert [r["track"]["id"] for r in search_case(service, "a", model, "red")["results"]] == ["a2"]


def test_zero_and_wrong_dimension_vectors_are_rejected():
    service, model = Service(), Model()
    index_case(service, "a", "reviewer", model)
    service.repo.rows[0]["vector"] = [0, 0]
    with pytest.raises(ValueError, match="Invalid CLIP"):
        search_case(service, "a", model, "person")
