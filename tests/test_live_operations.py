import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fedsight.core import digest
from fedsight.navigation import DEFAULT, pages, resolve
from fedsight.service import FedSight
from fedsight.services.live.archive import Archive, utc
from fedsight.services.live.bridge import materialize, preserve
from fedsight.services.live.insights import answer
from fedsight.services.live.live_pipeline import Pipeline, protected
from fedsight.services.live.stream_manager import StreamManager, site_record
from fedsight.theme import THEMES


def test_navigation_memory_and_shared_links():
    history = dict(DEFAULT, case="Video review", live="Live respond")
    assert resolve("case","live-wall",history) == ("live","Live wall")
    assert resolve("live","evidence-processing",history)==("case","Evidence & processing")
    assert resolve("live","blindsight",history)==("live","BlindSight")
    assert resolve("case",None,history)==("case","Video review")
    assert resolve("live","missing",history)==("live","Live respond")
    assert len(pages("live"))==9 and "Live wall" not in pages("case")


@pytest.mark.parametrize("mode",["case","live"])
def test_theme_contrast(mode):
    def luminance(hex_color):
        rgb = [int(hex_color[i:i+2],16)/255 for i in (1,3,5)]
        lin = [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(v*w for v,w in zip(lin,[.2126,.7152,.0722]))
    t = THEMES[mode]
    def check(fg,bg):
        a,b=sorted([luminance(t[fg]),luminance(t[bg])])
        assert (b+.05)/(a+.05)>=4.5, (mode,fg,bg)
    for fg in ["text","muted","accent","critical","warn"]:
        for bg in ["bg-from","bg-to","sidebar","card"]:
            check(fg,bg)
    check("on-accent","accent")


def detector(frame):
    return np.array([[10,10,35,60]],dtype=float),np.array([.95])


def test_worker_reconnect_stop_and_dedup(tmp_path):
    s=FedSight(tmp_path)
    site=site_record(s.repo,"test")["id"]
    camera=s.register_camera(site,"Test","test")["id"]
    s.repo.put("rule",dict(case_id=site,label="Entry",type="count",threshold=1,severity="high"),"test")
    attempts=[]
    class Fake:
        def read(self):
            return True,np.zeros((64,96,3),np.uint8)
        def isOpened(self):
            return len(attempts)>1
        def release(self):
            pass
        def get(self,_):
            return 25
    def open_fake(*args):
        attempts.append(1)
        return Fake(),None
    m=StreamManager(s,detector,open_fake)
    spec=dict(camera_id=camera,kind="demo",source="fake")
    try:
        assert m.start(site,spec)
        assert not m.start(site,spec)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and len(m.archive.samples(site,camera,0,time.time()))<3:
            time.sleep(.1)
        assert m.snapshot(site,camera)["status"]=="Online"
        assert len(attempts)==2
        records=s.repo.list(site,"event")
        assert len(records)==1 and records[0]["source_type"]=="live"
        assert len(m.bus)==1 and not s.repo.list(None,"case")
        assert s.repo.verify()["ok"]
        m.stop(site,camera)
        for t in m.streams[(site,camera)].workers:
            t.join(2)
        assert m.snapshot(site,camera)["status"]=="Stopped"
    finally:
        m.close()


def test_clip_to_case_and_retention(tmp_path):
    s=FedSight(tmp_path)
    site=site_record(s.repo,"alice")["id"]
    cam=s.register_camera(site,"Exit","alice")["id"]
    a=Archive(s)
    p=Pipeline(detector)
    start=time.time()-15
    frame=np.random.default_rng(42).integers(0,255,(72,96,3),dtype=np.uint8)
    masked=protected(frame,True)
    for i in range(25):
        ts=start+i*.5
        tracks,_=p.step(frame,[],cam,ts)
        a.append(site,cam,ts,p.run,masked,tracks)
    stamp=start+6
    event=s.repo.put("event",dict(case_id=site,source_type="live",source_id=cam,camera_id=cam,stamp=stamp,time_utc=utc(stamp),
             run=p.run,simulated=True,masked=True,method="zone",rule="Exit entry",rule_id="exit-zone",severity="critical",
             track_id=tracks[0]["id"],start_time=stamp,end_time=stamp),"worker")
    summary=answer(s.repo,None,site,"Any intrusions in the last 10 minutes?",15)
    assert len(summary["rows"])==1 and "10 minutes" in summary["answer"]
    bundle=materialize(s,a,event)
    cid,sid,sec=preserve(s,event,bundle,"alice",title="Flag review",assignee="alice",status="Under Review")
    assert sec==pytest.approx(5)
    src=s.repo.get(cid,sid,"source")
    assert src["provenance"]["masked"] and src["provenance"]["live_event_id"]==event["id"]
    assert digest(s.read_asset(src["asset"]))==src["sha256"]
    assert len(s.active_tracks(cid))==1
    assert s.active_tracks(cid)[0]["preview_asset"]
    assert preserve(s,event,bundle,"alice",case_id=cid)==(cid,sid,sec)
    assert len(s.repo.list(cid,"source"))==1
    assert s.repo.verify()["ok"] and any(e["action"]=="live.open_as_case" for e in s.repo.audit_events())
    assert all(x["ok"] for x in s.verify_assets(cid,"alice"))
    a.cleanup(site,1,1)
    assert not a.samples(site,cam,0,time.time())
    with pytest.raises(ValueError,match="expired"):
        materialize(s,a,event)
    assert s.read_asset(src["asset"])==bundle["clip"]


def test_all_live_pages_with_no_case(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FEDSIGHT_DATA_DIR",str(tmp_path))
    app=AppTest.from_file(Path(__file__).parents[1]/"app.py",default_timeout=60)
    app.query_params["mode"]="live"
    app.run()
    assert not app.exception
    assert app.title[0].value=="Live wall"
    assert not any(s.label=="Active case" for s in app.selectbox)
    for page in pages("live"):
        app.radio(key="workspace").set_value(page).run()
        assert not app.exception and not app.error, (page,list(app.exception),list(app.error))
        assert app.title[0].value==page


def test_live_zone_tool_reports_only_current_named_zone(tmp_path):
    s=FedSight(tmp_path)
    site=site_record(s.repo,"test")["id"]
    cam=s.register_camera(site,"Camera A","test")["id"]
    s.repo.put("rule",dict(case_id=site,label="Exit",type="zone",box=[.2,.2,.6,.6],severity="medium",camera_id=cam),"test")
    tracks=[dict(id="inside",center=[.3,.4],attributes={"upper_color":"gray"}),
            dict(id="outside",center=[.1,.8],attributes={"upper_color":"blue"})]
    m=SimpleNamespace(snapshot=lambda site,camera:dict(status="Online",latest=(time.time(),None,tracks,[],None)))
    result=answer(s.repo,m,site,"Who is in the Exit zone now?",5)
    assert [r["Local track"] for r in result["rows"]]==["inside"]
    assert not answer(s.repo,m,site,"Who is in the Garage zone now?",5)["rows"]


def test_blind_gap_requires_observed_destination_coverage(tmp_path):
    from fedsight.services.live.insights import overdue
    s=FedSight(tmp_path)
    site=site_record(s.repo,"test")["id"]
    a,b=[s.register_camera(site,k,"test")["id"] for k in ("Door A","Door B")]
    s.add_transition(site,"test",from_camera=a,to_camera=b,min_s=2,max_s=5,provenance="Measured corridor")
    zone=s.add_coverage_region(site,"test",kind="blind",label="Corridor",box=[.1,.1,.6,.6],provenance="Plan")
    s.add_blind_portal(site,"test",camera_id=a,blind_zone_id=zone["id"],role="exit",box=[.7,.2,.99,.8],provenance="Door")
    rows={a:[(50,[dict(id="person_1",center=[.8,.5])],None),(51,[],None)],b:[]}
    archive=SimpleNamespace(samples=lambda site,c,start,end:rows[c])
    assert not overdue(s.repo,archive,site,60)
    rows[b]=[(i,[],None) for i in range(52,56)]
    assert len(overdue(s.repo,archive,site,60))==1
    rows[b][1]=(53,[dict(id="other",center=[.3,.4])],None)
    assert not overdue(s.repo,archive,site,60)



def test_case_finding_creates_disabled_live_rule(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from fedsight.demo import create_demo
    from fedsight.services.live.stream_manager import active_rules
    s=FedSight(tmp_path)
    cid=create_demo(s)
    monkeypatch.setenv("FEDSIGHT_DATA_DIR",str(tmp_path))
    app=AppTest.from_file(Path(__file__).parents[1]/"app.py",default_timeout=60)
    app.query_params["page"]="video-review"
    app.run()
    app.radio(key="video_review_mode").set_value("Single recording").run()
    next(b for b in app.button if b.label=="Watch for this").click().run()
    next(b for b in app.button if b.label=="Save draft live rule").click().run()
    assert not app.exception and app.title[0].value=="Rules & events"
    site=s.repo.list(None,"live_site")[0]["id"]
    draft=s.repo.list(site,"rule")[0]
    assert draft["origin_case"]==cid and draft["enabled"] is False
    assert not active_rules(s.repo,site)
    app.toggle(key="enable_live_"+draft["id"]).set_value(True)
    next(b for b in app.button if b.label=="Update rule").click().run()
    assert active_rules(s.repo,site)[0]["id"]==draft["id"]


def test_stream_pause_resume_and_should_poll(tmp_path):
    import streamlit as st
    from fedsight.operations import should_poll
    s = FedSight(tmp_path)
    site = site_record(s.repo, "test")["id"]
    camera = s.register_camera(site, "TestCam", "test")["id"]
    
    class FakeCap:
        def read(self):
            return True, np.zeros((64, 96, 3), np.uint8)
        def isOpened(self):
            return True
        def release(self):
            pass
        def get(self, _):
            return 25

    m = StreamManager(s, detector, lambda *args: (FakeCap(), None))
    spec = dict(camera_id=camera, kind="demo", source="fake")
    try:
        assert m.start(site, spec)
        time.sleep(0.3)
        assert m.is_active(site)
        assert not m.is_paused(site)

        # Pause streams
        m.pause_all(site)
        assert m.is_paused(site)
        assert not m.is_active(site)
        assert m.snapshot(site, camera)["status"] == "Paused"

        # Resume streams
        m.resume_all(site)
        assert not m.is_paused(site)
        assert m.is_active(site)

        # Test should_poll logic
        st.session_state["live_auto_refresh"] = False
        assert should_poll(site, m, 5) is None  # Auto-refresh off -> None

        st.session_state["live_auto_refresh"] = True
        assert should_poll(site, m, 5) == 5     # Streams active + auto-refresh on -> 5

        m.pause_all(site)
        assert should_poll(site, m, 5) is None  # Streams paused -> None even if auto-refresh is True

        m.stop_all(site)
        for t in m.streams[(site, camera)].workers:
            t.join(2)
        assert m.snapshot(site, camera)["status"] == "Stopped"
    finally:
        m.close()


