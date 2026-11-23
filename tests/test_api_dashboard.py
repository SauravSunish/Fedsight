import pytest
from pathlib import Path
from fedsight.service import FedSight
from fedsight.demo import create_demo


def test_api_auth_and_case_scoping(tmp_path):
    from fastapi.testclient import TestClient
    from fedsight.api import create_app
    service = FedSight(tmp_path)
    cid = create_demo(service)
    token = "a" * 40
    client = TestClient(create_app(service, token))
    assert client.get("/health").status_code == 200
    assert client.get("/cases").status_code == 401
    headers = {"Authorization": "Bearer " + token}
    assert client.get("/cases", headers=headers).json()[0]["id"] == cid
    assert client.get("/cases/missing/proposals", headers=headers).status_code == 404
    proposal = service.repo.active_proposals(cid)[0]
    result = client.post(f"/cases/{cid}/proposals/{proposal['id']}/reviews", headers=headers, json={"decision": "accept", "reason": "compared synthetic source records"})
    assert result.status_code == 201
    assert result.json()["actor"] == "local-api"


def test_all_dashboard_pages_render(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from fedsight.dashboard import PAGES
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    service = FedSight(tmp_path)
    create_demo(service)
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    assert not at.exception
    for page in PAGES:
        at.selectbox(key="global_cmd_palette").set_value(page).run()
        assert at.title[0].value == page
        assert not at.exception, f"{page}: {at.exception}"


def test_dashboard_create_case_and_review(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    button = next(b for b in at.button if b.label == "Load synthetic walkthrough")
    button.click().run()
    assert not at.exception
    at.sidebar.radio[0].set_value("ThreadSight review").run()
    next(t for t in at.text_area if t.label == "Decision reason (required)").set_value("Illustrations checked")
    next(b for b in at.button if b.label == "Record review").click().run()
    assert not at.exception
    service = FedSight(tmp_path)
    cid = service.repo.list(None, "case")[0]["id"]
    assert len(service.repo.list(cid, "review")) == 1


def test_quick_jump_opens_selected_sighting(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR", str(tmp_path))
    service = FedSight(tmp_path)
    cid = create_demo(service)
    track = next(t for t in service.active_tracks(cid)[:10] if t["start_time"] > 0)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    jump = app.selectbox(key="global_cmd_palette")
    jump.set_value(next(option for option in jump.options if track["id"][-8:] in option)).run()
    assert not app.exception
    assert app.selectbox(key="global_cmd_palette").value is None
    assert app.radio(key="video_review_mode").value == "Single recording"
    assert app.selectbox(key="video_review_source").value == track["source_id"]
    assert app.slider(key="vr_frame_" + track["source_id"]).value == int(track["start_time"] * 25)
