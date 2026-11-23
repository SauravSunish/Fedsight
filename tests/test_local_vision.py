import io
import json
from PIL import Image
import pytest

from fedsight.local_vision import search_previews


def test_local_vision_returns_only_confirmed_indexed_crops():
    img = io.BytesIO()
    Image.new("RGB", (40, 80), "blue").save(img, "PNG")
    class Service:
        def read_asset(self, asset, actor):
            return img.getvalue()
    class Tools:
        service = Service()
        sources = {"s": {"name": "camera"}}
        tracks = {str(i): {"id": str(i), "source_id": "s", "start_time": i, "preview_asset": {"id": str(i)}} for i in range(2)}
        def track_evidence(self, track):
            return {"id": track["id"], "track_id": track["id"]}
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": '{"results":[{"image":1,"verdict":"match","reason":"hair visible"},{"image":2,"verdict":"uncertain","reason":"obscured"}]}'}}
    def post(url, **kwargs):
        assert url.startswith("http://127.0.0.1:")
        assert kwargs["json"]["messages"][0]["images"]
        assert "long hair" in kwargs["json"]["messages"][0]["content"]
        return Response()
    result = search_previews("long haie", Tools(), "reviewer", post=post)
    assert [e["id"] for e in result["evidence"]] == ["0", "1"]
    assert result["evidence"][1]["visual_verdict"] == "uncertain"
    assert "1 uncertain" in result["answer"]
    assert "not confirmed" in result["answer"]


def test_local_vision_rejects_incomplete_response():
    # An incomplete assessment cannot be presented as a complete search.
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": '{"results":[]}'}}
    class Service:
        def read_asset(self, asset, actor):
            out = io.BytesIO(); Image.new("RGB", (20, 20)).save(out, "PNG"); return out.getvalue()
    class Tools:
        service = Service()
        sources = {"s": {"name": "camera"}}
        tracks = {"1": {"id": "1", "source_id": "s", "start_time": 0, "preview_asset": {"id": "1"}}}
    with pytest.raises(ValueError, match="did not assess"):
        search_previews("long hair", Tools(), "reviewer", post=lambda *a, **k: Response())


def test_visual_search_runs_in_background(monkeypatch):
    import threading
    import time
    from fedsight.local_vision import VisionQueue
    started, release = threading.Event(), threading.Event()
    def slow_search(*args):
        started.set()
        assert release.wait(3)
        return {"answer": "done"}
    monkeypatch.setattr("fedsight.local_vision.search_previews", slow_search)
    queue = VisionQueue()
    task = queue.start(("root", "case", "session"), "hair", type("Tools", (), {"case_id": "case"})(), "reviewer")
    assert started.wait(1)
    assert task.snapshot()["status"] == "running"
    release.set()
    deadline = time.monotonic() + 3
    while task.snapshot()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(.01)
    assert task.snapshot()["result"] == {"answer": "done"}


def test_preview_selection_samples_each_camera_before_repeating():
    class Service:
        def read_asset(self, asset, actor):
            out = io.BytesIO(); Image.new("RGB", (20, 20)).save(out, "PNG"); return out.getvalue()
    class Tools:
        service = Service()
        sources = {"a": {"name": "camera a"}, "b": {"name": "camera b"}}
        tracks = {str(i): {"id": str(i), "source_id": "a" if i < 4 else "b",
                            "start_time": i, "preview_asset": {"id": str(i)}} for i in range(5)}
        def track_evidence(self, track):
            return {"id": track["id"], "track_id": track["id"]}
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": '{"results":[{"image":1,"verdict":"match","reason":"a"},{"image":2,"verdict":"match","reason":"b"}]}'}}
    result = search_previews("hair", Tools(), "reviewer", limit=2, post=lambda *a, **k: Response())
    assert [e["id"] for e in result["evidence"]] == ["0", "4"]


def test_rejected_previews_remain_available_for_manual_review():
    class Service:
        def read_asset(self, asset, actor):
            out = io.BytesIO(); Image.new("RGB", (20, 20)).save(out, "PNG"); return out.getvalue()
    class Tools:
        service = Service()
        sources = {"s": {"name": "camera"}}
        tracks = {"1": {"id": "1", "source_id": "s", "start_time": 0, "preview_asset": {"id": "1"}}}
        def track_evidence(self, track):
            return {"id": track["id"], "track_id": track["id"]}
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": '{"results":[{"image":1,"verdict":"no_match","reason":"hair unclear"}]}'}}
    result = search_previews("long hair", Tools(), "reviewer", post=lambda *a, **k: Response())
    assert not result["evidence"] and not result["working_set"]
    assert result["reviewed_no_match"][0]["visual_verdict"] == "no_match"


def test_local_vision_batches_over_twelve_previews():
    class Service:
        def read_asset(self, asset, actor):
            out = io.BytesIO()
            Image.new("RGB", (20, 20)).save(out, "PNG")
            return out.getvalue()
    class Tools:
        service = Service()
        sources = {"s": {"name": "camera"}}
        tracks = {str(i): {"id": str(i), "source_id": "s", "start_time": i, "preview_asset": {"id": str(i)}} for i in range(25)}
        def track_evidence(self, track):
            return {"id": track["id"], "track_id": track["id"]}
    calls = []
    class Response:
        def __init__(self, count):
            self.count = count
        def raise_for_status(self): pass
        def json(self):
            results = [{"image": i + 1, "verdict": "match" if i == 0 else "no_match", "reason": "test"} for i in range(self.count)]
            return {"message": {"content": json.dumps({"results": results})}}
    def mock_post(url, **kwargs):
        # Infer batch size from prompt text
        content = kwargs["json"]["messages"][0]["content"]
        import re
        m = re.search(r"Examine all (\d+) numbered crops", content)
        count = int(m.group(1)) if m else 12
        calls.append(count)
        return Response(count)
    result = search_previews("long hair", Tools(), "reviewer", limit=25, post=mock_post)
    assert calls == [12, 12, 1]
    # Each batch matched image 1 (i.e. tracks 0, 12, 24)
    assert [e["id"] for e in result["evidence"]] == ["0", "12", "24"]


def test_local_vision_connection_error_raises_clean_message():
    import requests
    class Service:
        def read_asset(self, asset, actor):
            out = io.BytesIO()
            Image.new("RGB", (20, 20)).save(out, "PNG")
            return out.getvalue()
    class Tools:
        service = Service()
        sources = {"s": {"name": "camera"}}
        tracks = {"1": {"id": "1", "source_id": "s", "start_time": 0, "preview_asset": {"id": "1"}}}
        def track_evidence(self, track):
            return {"id": track["id"], "track_id": track["id"]}
    def failing_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection refused")
    with pytest.raises(RuntimeError, match="Ollama is not running locally"):
        search_previews("long hair", Tools(), "reviewer", post=failing_post)

