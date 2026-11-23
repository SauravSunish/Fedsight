from pathlib import Path

import pytest

from fedsight.service import FedSight


def test_case_removal_is_scoped_and_keeps_checkpoint_valid(tmp_path):
    service = FedSight(tmp_path)
    repo = service.repo
    first, second = [repo.create_case("Same title", "", "test") for _ in range(2)]
    cid = first["id"]
    a = service.ingest(cid, "hall.mp4", b"sample", "test")["asset"]
    b = service.ingest(second["id"], "hall.mp4", b"sample", "test")["asset"]
    repo.put("investigator_message", {"case_id": cid, "text": "sample question"}, "test")
    repo.accept_message("node", cid, 1, "msg", "test")
    checkpoint = repo.checkpoint()
    repo.edit_case(cid, "Updated", "details", "retention", "test")
    assert repo.get(cid, cid)["description"] == "details"
    assert service.delete_case(cid, "test") == 0
    assert not (service.vault.root / a["storage_name"]).exists()
    assert service.read_asset(b) == b"sample"
    assert repo.list(None, "case") == [second]
    assert repo.storage_info(cid)["counts"] == {}
    with repo.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM replay WHERE case_id=?", (cid,)).fetchone()[0] == 0
    with pytest.raises(ValueError):
        repo.put("case", first, "stale worker", replace=True)
    with pytest.raises(ValueError):
        service.save_asset(b"late result", cid, "stale worker", role="preview")
    assert len(list(service.vault.root.glob("*.aes"))) == 1
    repo.compact()
    assert repo.verify(checkpoint)["ok"]


def test_busy_case_and_retry_locked_file(tmp_path, monkeypatch):
    from fedsight.processing_queue import QUEUE, ProcessingTask
    service = FedSight(tmp_path)
    cid = service.repo.create_case("Upload", "", "test")["id"]
    asset = service.ingest(cid, "hall.mp4", b"video", "test")["asset"]
    key = (str(tmp_path.resolve()), cid)
    task = ProcessingTask(cid)
    monkeypatch.setitem(QUEUE.tasks, key, task)
    with pytest.raises(ValueError, match="background task"):
        service.delete_case(cid, "test")
    task.update(status="completed")
    def locked(*args):
        raise PermissionError("File in use")
    monkeypatch.setattr(service.vault, "remove", locked)
    assert service.delete_case(cid, "test") == 1
    assert key not in QUEUE.tasks
    reopened = FedSight(tmp_path)
    assert reopened.repo.storage_info()["pending_files"] == 1
    assert reopened.clean_deleted_assets() == 0
    assert not (reopened.vault.root / asset["storage_name"]).exists()


def test_case_manager_create_edit_and_delete(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    def button(label):
        return next(b for b in app.button if b.label == label)
    button("Manage cases").click().run()
    next(t for t in app.text_input if t.label == "Case title").set_value("Disposable")
    button("Create case").click().run()
    assert not app.exception
    next(t for t in app.text_input if t.label == "Case name").set_value("Renamed")
    button("Save case details").click().run()
    assert FedSight(tmp_path).repo.list(None, "case")[0]["title"] == "Renamed"
    assert button("Permanently delete case").disabled
    next(c for c in app.checkbox if c.label.startswith("I want to permanently")).check().run()
    button("Permanently delete case").click().run()
    assert not app.exception
    assert FedSight(tmp_path).repo.list(None, "case") == []
    assert any(i.value == "Case deleted." for i in app.success)
    assert button("Create case")
