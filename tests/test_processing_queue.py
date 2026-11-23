import threading
import time

import pytest

from fedsight.embeddings import OSNetConfig
from fedsight.processing_queue import ProcessingQueue
from fedsight.service import FedSight
from fedsight.video import AnalyzeConfig


def test_cancelled_source_never_commits_partial_tracks(tmp_path, monkeypatch):
    import fedsight.video
    service = FedSight(tmp_path)
    case = service.repo.create_case("case", "", "tester")
    cid = case["id"]
    service.ingest(cid, "clip.mp4", b"video", "tester")
    def fake_analysis(*args, **kwargs):
        return [{"track_id": "partial"}], {"stop_reason": "cancelled", "incomplete": True}
    monkeypatch.setattr(fedsight.video, "analyze_video", fake_analysis)
    result = service.process(cid, "tester", AnalyzeConfig(), OSNetConfig(), should_stop=lambda: False)
    assert result["cancelled"] is True
    assert service.repo.list(cid, "track") == []
    assert service.repo.list(cid, "job")[-1]["status"] == "cancelled"
    assert service.repo.get(cid, cid, "case")["status"] == "Failed"
    assert service.repo.verify()["ok"]


def test_queue_prevents_duplicate_work_and_cancels(tmp_path):
    started, release = threading.Event(), threading.Event()
    class FakeService:
        root = tmp_path
        def process(self, case_id, actor, video, embedding, *, progress, force, should_stop, stage):
            started.set()
            release.wait(5)
            return {"cancelled": should_stop(), "failed_sources": [], "tracks": 0}
    queue = ProcessingQueue()
    service = FakeService()
    task = queue.start(service, "case", "tester", None, None)
    assert started.wait(2)
    with pytest.raises(ValueError):
        queue.start(service, "case", "tester", None, None)
    assert queue.cancel(str(tmp_path.resolve()), "case")
    release.set()
    deadline = time.monotonic() + 3
    while task.snapshot()["status"] not in {"cancelled", "failed", "completed"} and time.monotonic() < deadline:
        time.sleep(.01)
    assert task.snapshot()["status"] == "cancelled"


def test_resume_retries_cancelled_reprocess_but_keeps_prior_tracks(tmp_path, monkeypatch):
    import fedsight.video
    service = FedSight(tmp_path)
    cid = service.repo.create_case("case", "", "tester")["id"]
    source = service.ingest(cid, "clip.mp4", b"video", "tester")
    service.repo.put("job", {"case_id": cid, "source_id": source["id"], "status": "completed"}, "tester")
    service.repo.put("job", {"case_id": cid, "source_id": source["id"], "status": "cancelled"}, "tester")
    called = []
    def fake_analysis(*args, **kwargs):
        called.append(True)
        return [], {"stop_reason": "end_of_video", "incomplete": False, "frame_size": [100, 100]}
    monkeypatch.setattr(fedsight.video, "analyze_video", fake_analysis)
    result = service.process(cid, "tester", AnalyzeConfig(), OSNetConfig())
    assert called and not result["failed_sources"]
    assert service.repo.list(cid, "job")[-1]["status"] == "completed"
    assert service.repo.verify()["ok"]
