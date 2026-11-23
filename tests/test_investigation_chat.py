import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fedsight.core import TimeAnchor
from fedsight.demo import create_demo
from fedsight.investigation_chat import evidence_time, local_candidates, query_filters, vision_rerank
from fedsight.service import FedSight


def test_description_filters_and_source_linked_times():
    assert query_filters("person with blue shirt and red pants near 01:20")["upper"] == "blue"
    assert query_filters("person with blue shirt and red pants near 01:20")["lower"] == "red"
    source = {"id": "s", "name": "Hall Camera", "time_anchor": {}}
    tracks = [{"id": "a", "track_id": "a", "source_id": "s", "object_class": "person",
               "start_time": 70., "end_time": 90., "duration_s": 20., "attributes": {"upper_color": "blue", "lower_color": "red"},
               "embedding_meta": {"mean_quality": .8}},
              {"id": "b", "track_id": "b", "source_id": "s", "object_class": "person",
               "start_time": 70., "end_time": 90., "duration_s": 20., "attributes": {"upper_color": "green", "lower_color": "red"}}]
    found, _ = local_candidates("person with blue shirt and red pants near 01:20", tracks, {"s": source})
    assert [f["track"]["id"] for f in found] == ["a"]
    tracks[0]["crop_frames"], tracks[0]["crop_scores"] = [710, 790], [.2, .9]
    job = {"meta": {"fps": 10}}
    source["time_anchor"] = TimeAnchor("2026-01-01T00:00:00Z", "UTC", 0, 2, 0, "verified").__dict__
    when = evidence_time(tracks[0], source, job)
    assert when["preview_frame"] == 790
    assert when["preview_clip_s"] == 79
    assert when["utc_midpoint"].endswith("+00:00")
    assert query_filters("black backpack")["colour"] is None
    broad = [{**tracks[0], "id": f"t{i}", "track_id": f"t{i}"} for i in range(90)]
    assert len(local_candidates("person", broad, {"s": source})[0]) == 90


def test_vision_response_ids_are_validated_and_originals_stay_local(tmp_path):
    image = Image.fromarray(np.full((80, 40, 3), 120, dtype=np.uint8))
    output = io.BytesIO(); image.save(output, format="PNG")
    class Service:
        def read_asset(self, asset, actor): return output.getvalue()
    class Responses:
        kwargs = None
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"output_text": json.dumps({"matches": [
                {"image_id": 1, "verdict": "match", "visible_reason": "blue clothing"},
                {"image_id": 99, "verdict": "match", "visible_reason": "fabricated ID"}]})})()
    class Client:
        responses = Responses()
    candidates = [{"track": {"id": "t", "preview_asset": {"asset_id": "p"}}, "score": .5,
                   "verification": "metadata_candidate", "reasons": []}]
    client = Client()
    matches = vision_rerank("blue jacket", candidates, Service(), "tester", "", client=client)
    assert len(matches) == 1 and matches[0]["track"]["id"] == "t"
    assert client.responses.kwargs["store"] is False
    content = client.responses.kwargs["input"][0]["content"]
    assert len([c for c in content if c["type"] == "input_image"]) == 1
    assert all("original" not in c.get("text", "").lower() for c in content)


def test_chatgpt_plan_vision_uses_required_streaming_flags():
    image = Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8))
    output = io.BytesIO(); image.save(output, format="PNG")
    class Service:
        def read_asset(self, asset, actor): return output.getvalue()
    class Responses:
        kwargs = None
        def create(self, **kwargs):
            self.kwargs = kwargs
            completed = type("Result", (), {"output_text": json.dumps({"matches": [
                {"image_id": 1, "verdict": "uncertain", "visible_reason": "partly obscured"}]})})()
            return iter([type("Event", (), {"type": "response.completed", "response": completed})()])
    class Client:
        responses = Responses()
    client = Client()
    candidate = {"track": {"id": "t", "preview_asset": {"asset_id": "p"}}, "score": .5,
                 "verification": "metadata_candidate", "reasons": []}
    found = vision_rerank("blue coat", [candidate], Service(), "tester", "", client=client, stream=True)
    assert found[0]["verification"] == "uncertain"
    assert client.responses.kwargs["stream"] is True
    assert client.responses.kwargs["store"] is False


def test_investigation_chat_page_renders_without_api_key(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = FedSight(tmp_path)
    create_demo(service)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.sidebar.radio[0].set_value("FedSight Investigator").run()
    assert not app.exception
    app.chat_input[0].set_value("person with red shirt").run()
    assert not app.exception
    searches = service.repo.list(service.repo.list(None, "case")[0]["id"], "chat_search")
    assert len(searches) == 1
    assert searches[0]["engine"] == "local_metadata"
    assert len(service.repo.list(service.repo.list(None, "case")[0]["id"], "investigator_message")) == 1


def test_investigator_evidence_jumps_to_video_review(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    service = FedSight(tmp_path)
    create_demo(service)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.sidebar.radio[0].set_value("FedSight Investigator").run()
    engine = next(item for item in app.selectbox if item.label == "Choose engine")
    engine.set_value("Built-in (offline)").run()
    app.chat_input[0].set_value("show the timeline").run()
    assert not app.exception
    jump = next(button for button in app.button if button.label == "Jump to clip")
    jump.click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Video review"
