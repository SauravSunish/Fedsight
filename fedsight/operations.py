"""Streamlit live operations and the two explicit case handoffs."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import io
import base64
import time
import wave

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .core import uid
from .navigation import REGISTRY
from .rules import validate_rule
from .services.live.archive import image, utc
from .services.live.bridge import materialize, preserve
from .services.live.live_pipeline import Cooldown, inside, matched, rendered
from .services.live.stream_manager import SAMPLES, StreamManager, active_rules, site_record
from .theme import palette, plot, tokens

SEVERITIES = ("low","medium","high","critical")
STATES = ("New","Acknowledged","False positive","Escalated")
GUIDANCE = {
    "Live wall":"Start demo streams to replay local recordings. Select a camera to enlarge its latest frame and review flags from the last minute.",
    "Live respond":"Filter incoming flags and inspect the timestamped footage. Acknowledge a flag or preserve its clip in a case before the rolling buffer expires.",
    "Rules & events":"Choose an observable condition and camera. Test it on buffered samples, then enable it for incoming frames. Schedules use UTC.",
    "Camera intelligence":"Connect a replay, webcam or RTSP stream to a mapped camera. Configure its position and directed travel times under Camera map and routes.",
    "Spatial intelligence":"Inspect camera status on the site map and configure a floor plan. Calibration relates image points to positions on that plan.",
    "BlindSight":"Map blind areas and review overdue transitions. A gap is a timing hypothesis that needs human review.",
    "Live analytics":"Inspect sampled occupancy, line crossings and rule frequency. Each heatmap uses one camera’s image coordinates.",
    "Live Investigator":"Ask about rule flags in a time window or current occupancy in a named zone. Expand Evidence used to check the recorded times and camera references.",
    "System health":"Check capture, detector latency and queue drops. Restart an individual source or adjust the site’s rolling footage budget.",
}


@st.cache_resource
def manager(root):
    from .service import FedSight
    return StreamManager(FedSight(Path(root)))


def configs(repo,site):
    return {c["camera_id"]:c for c in repo.list(site,"stream_config")}


def cameras_by_id(repo,site):
    return {c["id"]:c for c in repo.list(site,"camera")}


def events(repo,site):
    with repo.connect() as db:
        revision = db.execute("SELECT max(rowid) FROM records WHERE case_id=? AND kind IN ('event','event_review')",
                              (site,)).fetchone()[0]
    return _event_view(str(repo.path.resolve()), site, revision, repo)


@st.cache_data(max_entries=12,ttl=30,show_spinner=False)
def _event_view(database, site, revision, _repo):
    decisions = {r["event_id"]:r["decision"] for r in _repo.list(site,"event_review")}
    return [{**e,"state":decisions.get(e["id"],"New")} for e in reversed(_repo.list(site,"event"))]


def should_poll(site, m, interval):
    """Only auto-poll if user enabled live auto-refresh AND streams are actively running."""
    if not st.session_state.get("live_auto_refresh", False):
        return None
    if not m.is_active(site):
        return None
    return interval


def site_picker(service, actor):
    repo = service.repo
    sites = {s["id"]:s for s in repo.list(None,"live_site")}
    if not sites:
        s = site_record(repo,actor)
        sites[s["id"]] = s
    if st.session_state.get("site_memory") in sites and "ops_site" not in st.session_state:
        st.session_state.ops_site = st.session_state.site_memory
    pending = st.session_state.pop("next_site", None)
    if pending in sites:
        st.session_state.ops_site = pending
    site = st.selectbox("Site / camera group",sites.keys(),format_func=lambda k:sites[k]["label"],key="ops_site")
    st.session_state.site_memory = site
    with st.expander("Add a site"):
        with st.form("new_site",clear_on_submit=True):
            label = st.text_input("Site name")
            if st.form_submit_button("Create site") and label.strip():
                s = site_record(repo,actor,label,uid("site"))
                st.session_state.site_memory = s["id"]
                st.session_state["next_site"] = s["id"]
                st.rerun()
    m = manager(str(service.root))
    cams = configs(repo,site)
    states = [m.snapshot(site,k) for k in cams]
    online_count = sum(s["status"]=="Online" for s in states)
    paused_count = sum(s["status"]=="Paused" for s in states)
    any_running = online_count > 0
    any_paused = paused_count > 0

    # Tactical Stream Resource & Execution Controls
    if cams:
        status_label = "🟢 STREAMS ACTIVE" if any_running else ("🟡 STREAMS PAUSED" if any_paused else "⚪ STREAMS IDLE")
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:750;letter-spacing:0.06em;color:var(--muted);margin:4px 0 2px 0;">'
            f'{status_label}'
            f'</div>',
            unsafe_allow_html=True
        )
        ctrl_c1, ctrl_c2 = st.columns(2)
        if any_running:
            with ctrl_c1:
                if st.button("⏸ Pause", key="btn_pause_streams", help="Pause video capture and neural inference to save 100% CPU/GPU resources", use_container_width=True):
                    m.pause_all(site)
                    st.rerun()
            with ctrl_c2:
                if st.button("⏹ Stop", key="btn_stop_streams", help="Stop all background stream workers completely", use_container_width=True):
                    m.stop_all(site)
                    st.rerun()
        elif any_paused:
            with ctrl_c1:
                if st.button("▶ Resume", key="btn_resume_streams", help="Resume live camera feeds and analysis", use_container_width=True):
                    m.resume_all(site)
                    st.rerun()
            with ctrl_c2:
                if st.button("⏹ Stop", key="btn_stop_streams", help="Stop all background stream workers completely", use_container_width=True):
                    m.stop_all(site)
                    st.rerun()
        else:
            with ctrl_c1:
                if st.button("▶ Start", key="btn_start_streams", help="Start configured camera streams", use_container_width=True):
                    for cfg in cams.values():
                        m.start(site, cfg)
                    st.rerun()
            with ctrl_c2:
                if st.button("↻ Refresh", key="btn_refresh_site", help="Refresh stream telemetry", use_container_width=True):
                    st.rerun()

    # User-controlled background polling (off by default so it never runs unprompted)
    st.checkbox("Live background refresh", value=False, key="live_auto_refresh",
                help="Automatically poll and re-render live feeds in the background. Leave unchecked to completely stop background polling.")

    @st.fragment(run_every=should_poll(site, m, 5))
    def status():
        cams = configs(repo,site)
        states = [m.snapshot(site,k) for k in cams]
        last = m.last_flag(site) or "—"
        online = sum(s['status']=='Online' for s in states)
        paused = sum(s['status']=='Paused' for s in states)
        if paused > 0 and online == 0:
            status_text = f"{paused}/{len(cams)} streams paused (standby)"
        elif online > 0:
            status_text = f"{online}/{len(cams)} streams online"
        else:
            status_text = f"0/{len(cams)} streams active (idle)"
        st.caption(f"{status_text} · "
                   f"{max((s['lag'] for s in states),default=0):.1f}s lag · {m.archive.usage(site)/1048576:.1f} MB buffered")
        st.caption(f"Last flag UTC · {last[11:19] if last != '—' else last}")
    status()
    return sites[site]


def start_demo(service,site,actor,m):
    files = sorted(SAMPLES.glob("epfl*.mp4"))[:2] or sorted(SAMPLES.rglob("*.mp4"))[:2]
    if not files:
        raise ValueError("Add a demo MP4 to data/sample_videos or upload a replay.")
    repo = service.repo
    for i,path in enumerate(files):
        name = path.relative_to(SAMPLES).as_posix()
        with repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            saved = [repo._decode(r) for r in db.execute("SELECT * FROM records WHERE case_id=? AND kind='stream_config'",(site,))]
            config = next((s for s in saved if s["kind"]=="demo" and s["source"]==name),None)
            if not config:
                camera = repo._put(db,"camera",dict(case_id=site,label=f"Demo camera {i+1}",x=.2+.5*i,y=.5,zone="",site="Demo replay"),actor)
                config = repo._put(db,"stream_config",dict(case_id=site,camera_id=camera["id"],source=name,kind="demo",masked=False),actor)
                repo._put(db,"rule",dict(case_id=site,type="count",threshold=1,label="Person visible",severity="low",camera_id=camera["id"]),actor)
        m.start(site,config)


def live_page(service, site_record_, actor, page, beginner):
    site = site_record_["id"]
    m = manager(str(service.root))
    @st.fragment(run_every=should_poll(site, m, 5))
    def header():
        cfg = configs(service.repo,site)
        online = [k for k in cfg if m.snapshot(site,k)["status"]=="Online"]
        st.markdown(f'<div class="tactical-ribbon"><strong>● LIVE</strong><span>{escape(site_record_["label"])}</span>'
                    f'<span>{len(online)} cameras online</span><span>Updated {utc(time.time())[11:19]} UTC</span>'
                    f'<span>Reviewer · {escape(actor)}</span></div>',unsafe_allow_html=True)
        kinds = {v["kind"] in ("demo","upload") for v in cfg.values()}
        if kinds == {True,False}:
            st.info("MIXED STREAMS — real cameras and recorded replays are labelled on each tile. Alerts are rule-based flags.")
        elif kinds == {False}:
            st.info("CAMERA STREAMS — configured cameras supply live frames. Alerts are rule-based flags, not identity matches.")
        else:
            st.info("SIMULATED STREAMS — live mode replays recorded footage as if it were live. Alerts are rule-based flags, not identity matches.")
    header()
    if m.is_paused(site):
        st.warning("⏸ Live camera streams are currently PAUSED (Zero Resource Mode). Real-time capture and neural inference are suspended. Click 'Resume' in the sidebar to activate.")
    pending = st.session_state.pop("open_live_evidence", None)
    if pending and pending[0] == site:
        bridge_dialog(service, site, service.repo.get(site,pending[1],"event"), actor, existing=pending[2])
    if beginner:
        with st.expander("How to use "+page):
            st.write(GUIDANCE[page])
    if not configs(service.repo,site) or page=="Live wall":
        if st.button("Start demo streams",type="primary"):
            start_demo(service,site,actor,m)
            st.rerun()
    try:
        route = REGISTRY[page]
        handler = globals()[route.live_handler or route.handler]
        handler(service,site,actor,m)
    except Exception as exc:
        st.error(f"Live workspace error ({page}): {exc}")


def jump(mode,page):
    st.session_state.route_request = (mode,page)


def request_bridge(site, event, existing=False):
    # Open the dialog from the full application so tile/feed fragment refreshes
    # cannot replace it while the operator is filling in the case details.
    st.session_state.open_live_evidence = (site, event["id"], existing)
    st.rerun()


@st.dialog("Preserve live evidence")
def bridge_dialog(service,site,e,actor,existing=False):
    editable = {c["id"]:c["title"] for c in service.repo.list(None,"case") if not c["demo"] and c["status"] in {"Created","Under Review","Failed"}}
    cid = st.selectbox("Destination case",editable,format_func=editable.get) if existing and editable else None
    if existing and not cid:
        st.info("Create or reopen an editable case first.")
        return
    title = st.text_input("Case name",value=e["rule"]+" · "+e["time_utc"][:19]) if not existing else ""
    assignee = st.text_input("Assignee",actor) if not existing else actor
    status = st.selectbox("Opening status",["Created","Under Review"]) if not existing else "Created"
    st.caption("Preserves a 10-second timestamped clip, event snapshot, observed tracks, crops and capture provenance.")
    if st.button("Preserve and open Video review",type="primary"):
        try:
            bundle = materialize(service,manager(str(service.root)).archive,e)
            cid,sid,sec = preserve(service,e,bundle,actor,case_id=cid,title=title,assignee=assignee,status=status)
            st.session_state.open_case = cid
            st.session_state.investigator_jump = dict(case_id=cid,source_id=sid,seconds=sec)
            jump("case","Video review")
            st.rerun()
        except (ValueError,OSError) as exc:
            st.error(str(exc))


def decision(repo,site,e,actor,state):
    repo.put("event_review",dict(case_id=site,event_id=e["id"],decision=state,actor=actor,source_type="live"),actor)


def recent_events(service,actor):
    sites = service.repo.list(None,"live_site")
    if not sites:
        return
    with st.expander("Recent live events"):
        recent = sorted((e for s in sites for e in events(service.repo,s["id"])[:3]),key=lambda e:e["stamp"],reverse=True)[:3]
        if not recent:
            st.caption("No live flags recorded yet.")
        for e in recent:
            st.write(f"{e['time_utc'][:19]} UTC · {e['rule']} · {e['severity']}")
            if st.button("Open as case",key="recent_"+e["id"]):
                bridge_dialog(service,e["case_id"],e,actor)


def watch_editor(service,track,actor):
    sites = service.repo.list(None,"live_site") or [site_record(service.repo,actor)]
    site = st.selectbox("Live site",sites,format_func=lambda s:s["label"])["id"]
    cams = cameras_by_id(service.repo,site)
    cam = st.selectbox("Watch on camera",[None,*cams],format_func=lambda k:cams[k]["label"] if k else "All site cameras")
    color = track.get("attributes",{}).get("upper_color")
    kind = st.selectbox("Observable condition",["Upper clothing colour","Zone entry"] if color and color!="unknown" else ["Zone entry"])
    fields = dict(type="appearance",upper_color=color) if kind=="Upper clothing colour" else zone_inputs()
    hours = st.slider("Active hours (UTC)",0,24,(0,24))
    label = st.text_input("Rule name",value=f"Watch for {color} upper clothing" if kind=="Upper clothing colour" else "Watch zone entry")
    if st.button("Save draft live rule",type="primary"):
        if not label.strip() or hours[0] == hours[1]:
            raise ValueError("Enter a rule name and a non-empty time window.")
        validate_live(fields)
        draft = service.repo.put("rule",dict(case_id=site,enabled=False,severity="medium",label=label,camera_id=cam,hours=hours,
                    origin_case=track["case_id"],origin_track=track["id"],**fields),actor)
        st.session_state.pending_rule = (site,draft["id"])
        st.session_state.next_site = site
        st.session_state.pop("watch_finding",None)
        jump("live","Rules & events")
        st.rerun()


def watch_button(service,track,actor,key):
    target=f"watch_{key}_{track['id']}"
    st.button("Watch for this",key=target,on_click=lambda:st.session_state.update(watch_finding=target))
    if st.session_state.get("watch_finding")==target:
        with st.container(border=True):
            st.subheader("Draft a live rule")
            watch_editor(service,track,actor)
            st.button("Close draft",on_click=lambda:st.session_state.pop("watch_finding",None))


def zone_inputs():
    bounds = [col.number_input(label,0.,1.,default) for col,label,default in zip(st.columns(4),("Left","Top","Right","Bottom"),(.2,.2,.8,.8))]
    return dict(type="zone",box=bounds)


def validate_live(fields):
    if fields["type"] != "appearance":
        validate_rule(fields)


def wall(service,site,actor,m):
    cfg = configs(service.repo,site)
    if not cfg:
        st.info("Connect an IP camera, USB webcam or replay to see it on the wall.")
        if st.button("Connect a camera →", key="wall_connect"):
            jump("live", "Camera intelligence")
            st.rerun()
        return
    cams = cameras_by_id(service.repo,site)
    size = st.select_slider("Wall tiles",options=[1,2,4,9],value=4)
    rate = st.selectbox("Refresh interval (seconds)",[1,2,5],index=1)
    keys = st.multiselect("Cameras on wall",list(cfg),default=list(cfg)[:9],format_func=lambda k:cams.get(k,{}).get("label",str(k)[-8:]),key="wall_cameras_"+site)
    with st.expander("Overlay controls"):
        options = {k:st.checkbox(label,True,key="live_overlay_"+k) for k,label in
                   (("boxes","Boxes"),("ids","Track IDs"),("trails","Motion trails"),("zones","Zones and lines"),("hud","HUD cards"))}
    
    col_w1, col_w2 = st.columns([3, 1])
    with col_w2:
        if st.button("↻ Refresh feeds", key="btn_refresh_wall", help="Fetch latest analyzed frames immediately", use_container_width=True):
            st.rerun()

    if m.is_paused(site):
        st.info("⏸ Camera streams are currently PAUSED (0 FPS · Standby). Resume from the sidebar to continue live surveillance.")

    @st.fragment(run_every=should_poll(site, m, rate))
    def tiles():
        focus_key = "live_focus_"+site
        focus = st.session_state.get(focus_key)
        pin_key = "pinned_camera_"+site
        pin = st.session_state.get(pin_key)
        order = ([pin] if pin in keys else [])+[k for k in keys if k!=pin]
        cols = st.columns(1 if focus or size==1 else (3 if size==9 else 2))
        for idx,k in enumerate([focus] if focus in cfg else order[:size]):
            state = m.snapshot(site,k)
            with cols[idx%len(cols)], st.container(border=True):
                label = cams.get(k,{}).get("label",str(k)[-8:])
                origin = "Replay" if cfg[k]["kind"] in ("demo","upload") else "Real camera"
                st.markdown(f"**{label}** · {origin}")
                st.caption(f"● {state['status']} · {state['fps']:.1f} analyzed FPS")
                if state["latest"]:
                    stamp,frame,tracks,rules_,_ = state["latest"]
                    try:
                        st.image(rendered(frame,tracks,rules_,stamp,options),channels="BGR",use_container_width=True)
                    except Exception:
                        st.image(frame,channels="BGR",use_container_width=True)
                else:
                    st.caption("Waiting for an analyzed frame.")
                e = state["last_event"]
                if e:
                    st.caption(f"Last flag · {e['rule']} · {e['time_utc'][11:19]} UTC")
                if not focus and st.button("Focus "+label,key="focus_"+k):
                    st.session_state[focus_key] = k
                    st.rerun(scope="app")
                if focus:
                    if st.button("Back to wall"):
                        st.session_state.pop(focus_key,None)
                        st.rerun()
                    if st.button("Unpin" if pin==k else "Pin camera"):
                        st.session_state[pin_key] = None if pin==k else k
                        st.rerun()
                    recent = [e for e in events(service.repo,site) if e["camera_id"]==k and e["stamp"]>=time.time()-60]
                    st.caption("Flags in the last 60 seconds")
                    if recent:
                        st.dataframe([{"UTC":e["time_utc"],"Rule":e["rule"],"Severity":e["severity"]} for e in recent],hide_index=True)
                    if e and st.button("Acknowledge latest alert"):
                        decision(service.repo,site,e,actor,"Acknowledged")
                        st.toast("Alert acknowledged")
                    if e and st.button("Open as case"):
                        request_bridge(site,e)
                    if state["latest"] and st.button("Preserve this camera selection"):
                        stamp,_,tracks,_,run = state["latest"]
                        selected = service.repo.put("event",dict(case_id=site,source_type="live",camera_id=k,source_id=m.source_id(site,k),stamp=stamp,
                            time_utc=utc(stamp),run=run,simulated=origin=="Replay",masked=cfg[k].get("masked",False),count=len(tracks),
                            method="selection",rule_id="manual",rule="Operator selection",severity="low",start_time=stamp,end_time=stamp),actor)
                        request_bridge(site,selected)
    tiles()


def notify_flags(records,site,sound):
    newest = next((e for e in records if e["severity"] in ("high","critical") and e["state"]=="New"),None)
    key = "seen_flag_"+site
    if newest and newest["id"] != st.session_state.get(key):
        st.session_state[key] = newest["id"]
        st.toast(newest["severity"].upper()+" · "+newest["rule"])
        if sound:
            buf = io.BytesIO()
            with wave.open(buf,"wb") as out:
                out.setparams((1,2,8000,0,"NONE","not compressed"))
                out.writeframes((5000*np.sin(2*np.pi*660*np.arange(1600)/8000)).astype("<i2").tobytes())
            st.audio(buf.getvalue(),format="audio/wav",autoplay=True)


def respond(service,site,actor,m):
    repo = service.repo
    cams = cameras_by_id(repo,site)
    cols = st.columns(4)
    severity = cols[0].multiselect("Severity",SEVERITIES)
    camera = cols[1].multiselect("Camera",cams,format_func=lambda k:cams[k]["label"])
    state = cols[2].multiselect("State",STATES)
    rule = cols[3].multiselect("Rule",sorted({r["label"] for r in repo.list(site,"rule")}))
    alerts = st.checkbox("Notify on high or critical alerts",True)
    sound = st.checkbox("Play alert sound",False,disabled=not alerts)
    @st.fragment(run_every=should_poll(site, m, 3))
    def feed():
        all_events = events(repo,site)
        if alerts:
            notify_flags(all_events,site,sound)
        filtered = [e for e in all_events if (not severity or e["severity"] in severity) and (not camera or e["camera_id"] in camera)
                    and (not state or e["state"] in state) and (not rule or e["rule"] in rule)]
        if not filtered:
            st.info("No flags match this view yet.")
            return
        page_count = (len(filtered)+19)//20
        page_no = st.selectbox("Flag history page",range(1,page_count+1),key="history_page_"+site) if page_count>1 else 1
        for e in filtered[(page_no-1)*20:page_no*20]:
            with st.container(border=True):
                l,r = st.columns([1,4])
                rows = m.archive.samples(site,e["camera_id"],e["stamp"],e["stamp"],pixels=True,run=e["run"])
                if rows:
                    stamp,tracks,blob = rows[0]
                    l.image(rendered(image(blob),tracks,[],stamp),channels="BGR")
                else:
                    l.caption("Preview expired")
                with r:
                    if e["severity"]=="critical":
                        st.markdown('<div role="alert" class="critical-alert"><strong>Critical alert</strong></div>',unsafe_allow_html=True)
                    st.markdown(f"**{e['rule']}** · {cams[e['camera_id']]['label']}")
                    st.caption(f"{e['time_utc']} · {e['severity']} · {e['state']} · {'Replay' if e['simulated'] else 'Real camera'}")
                with st.expander("Review flag · "+e["id"][-6:]):
                    if st.button("View 10-second clip",key="clip_"+e["id"]):
                        try:
                            bundle = materialize(service,m.archive,e)
                            st.session_state["live_detail"] = (site,e["id"],bundle["clip"])
                        except ValueError as exc:
                            st.info(str(exc))
                    detail = st.session_state.get("live_detail")
                    if detail and detail[:2] == (site,e["id"]) and rows:
                        st.video(detail[2])
                    for col,action,label in zip(st.columns(3),STATES[1:],("Acknowledge","False positive","Escalate")):
                        if col.button(label,key=action+e["id"],disabled=action==e["state"]):
                            decision(repo,site,e,actor,action)
                            st.rerun()
                    c1,c2 = st.columns(2)
                    if c1.button("Open as case",key="open_"+e["id"]):
                        request_bridge(site,e)
                    if c2.button("Add to existing case",key="add_"+e["id"]):
                        request_bridge(site,e,existing=True)
        if page_count>1:
            st.caption(f"Page {page_no} of {page_count} · {len(filtered)} matching flags")
    feed()
    with st.expander("Recent cases"):
        for c in reversed(repo.list(None,"case")[-5:]):
            if st.button(c["title"],key="live_case_"+c["id"]):
                st.session_state.open_case = c["id"]
                jump("case","Overview")
                st.rerun()


def new_rule(service, site, actor):
    repo = service.repo
    cams = cameras_by_id(repo,site)
    kind = st.selectbox("Condition",["zone","line","count","crowd","appearance"])
    with st.form("live_new_rule"):
        label = st.text_input("Rule name",placeholder="Exit zone entry")
        cam = st.selectbox("Apply to camera",[None,*cams],format_func=lambda k:cams[k]["label"] if k else "All cameras")
        severity = st.selectbox("Alert severity",SEVERITIES,index=1)
        if kind=="zone":
            fields = zone_inputs()
        elif kind=="line":
            fields = dict(type=kind,axis=st.selectbox("Axis",["x","y"]),position=st.slider("Line position",.01,.99,.5))
        elif kind=="appearance":
            fields = dict(type=kind,upper_color=st.selectbox("Upper clothing colour",["red","yellow","blue","green","black","white","gray","brown"]))
        else:
            fields = dict(type=kind,threshold=st.number_input("People in frame",1,64,3))
        a,b=st.columns(2)
        hours=[a.number_input("Start hour (UTC)",0,23,0),b.number_input("End hour (UTC)",0,24,24)]
        enabled=st.checkbox("Enable this new rule immediately",True)
        if st.form_submit_button("Save live rule"):
            if not label.strip() or hours[0]==hours[1]:
                raise ValueError("Enter a name and two different schedule endpoints.")
            validate_live(fields)
            repo.put("rule",dict(case_id=site,label=label.strip(),camera_id=cam,severity=severity,hours=hours,enabled=enabled,**fields),actor)
            st.rerun()


def rules(service,site,actor,m):
    repo = service.repo
    saved = {r["id"]: r for r in active_rules(repo,site,include_disabled=True)}
    with st.expander("Create a live rule", expanded=not saved):
        new_rule(service,site,actor)
    if saved:
        select_key = "managed_rule_"+site
        pending = st.session_state.pop("pending_rule", None)
        if pending and pending[0]==site and pending[1] in saved:
            st.session_state[select_key] = pending[1]
            st.success("Draft saved. Review the condition and schedule, then enable it to watch incoming frames.")
        rule_id = st.selectbox("Manage a rule",list(saved),key=select_key,
                               format_func=lambda key: saved[key]["label"]+" · "+key[-6:])
        r = saved[rule_id]
        with st.form("edit_"+r["id"]):
            enabled=st.toggle("Active in live",r.get("enabled",True),key="enable_live_"+r["id"])
            severity=st.selectbox("Severity",SEVERITIES,index=SEVERITIES.index(r["severity"]))
            a,b=st.columns(2)
            hours=[a.number_input("Start UTC",0,23,r.get("hours",[0,24])[0]),b.number_input("End UTC",0,24,r.get("hours",[0,24])[1])]
            if st.form_submit_button("Update rule"):
                if hours[0]==hours[1]:
                    raise ValueError("Choose different schedule endpoints.")
                repo.put("rule_state",dict(case_id=site,rule_id=r["id"],enabled=enabled,severity=severity,hours=hours),actor)
                st.rerun()
        if st.button("Test on last 5 minutes"):
            total = hits = 0
            end = time.time()
            for cam in configs(repo,site):
                before = {}
                cooldown = Cooldown()
                for stamp,tracks,_ in m.archive.samples(site,cam,end-300,end):
                    hits += sum(cooldown.accept(r["id"],tid,stamp) for tid,_ in matched(r,tracks,before,stamp,cam))
                    before = {t["id"]:t["center"] for t in tracks}
                    total += 1
            st.info(f"{hits} flags on {total} retained samples in this window.")
    log=events(repo,site)
    if log:
        st.dataframe([{k:e[k] for k in ("time_utc","rule","severity","state")} for e in log[:200]],hide_index=True)


def status_rows(service,site,m):
    cfg=configs(service.repo,site)
    cams=cameras_by_id(service.repo,site)
    calibrated={c.get("camera_id") for c in service.floor_calibrations(site)}
    result=[]
    for k,c in cams.items():
        s=m.snapshot(site,k)
        result.append({"Camera":c["label"],"Source":cfg.get(k,{}).get("kind","Unconfigured"),"Status":s["status"],
                       "Analyzed FPS":s["fps"],"Lag (s)":s["lag"],"Queue":s["queue"],"Dropped samples":s["dropped"],
                       "Latest UTC":utc(s["latest"][0]) if s["latest"] else "—",
                       "Calibration":"Configured" if k in calibrated else "Uncalibrated"})
    return result


def camera_page(service,site,actor,m):
    from .dashboard import cameras
    repo=service.repo
    cams=cameras_by_id(repo,site)
    with st.expander("Connect a camera or replay",expanded=True):
        kind=st.selectbox("Source type",["upload","rtsp","usb"],format_func={"upload":"Upload a replay","rtsp":"RTSP camera","usb":"USB webcam"}.get)
        if kind == "rtsp":
            st.caption("Enter the RTSP stream address supplied by your camera. The camera and laptop must be able to reach each other on the network.")
        with st.form("connect_stream",clear_on_submit=True):
            choice=st.selectbox("Mapped camera",[None,*cams],format_func=lambda k:cams[k]["label"] if k else "Register a new camera")
            name=st.text_input("New camera name")
            uploaded=st.file_uploader("Replay video",type=["mp4","avi","mov","mkv"]) if kind=="upload" else None
            source=st.text_input("RTSP address",type="password") if kind=="rtsp" else (st.number_input("USB camera index",0,7,0) if kind=="usb" else "")
            masked=st.checkbox("Privacy mask all pixels on this stream",help="Applies before live frames enter the buffer, tiles, snapshots and case exports.")
            if st.form_submit_button("Connect and start"):
                if kind=="rtsp" and not source.lower().startswith(("rtsp://","rtsps://")):
                    raise ValueError("Enter an RTSP URL.")
                if kind=="upload" and (not uploaded or not 0<uploaded.size<=200*1024**2):
                    raise ValueError("Select a video of up to 200 MB.")
                if not choice:
                    choice=service.register_camera(site,name,actor,x=0.,y=0.,zone="")["id"]
                spec=dict(kind=kind,source=source,masked=masked,case_id=site,camera_id=choice)
                if uploaded:
                    spec.update(source=Path(uploaded.name).name,asset=service.save_asset(uploaded.getvalue(),site,actor,role="stream_replay"))
                saved=repo.put("stream_config",spec,actor)
                m.restart(site,saved)
                st.session_state["camera_connected_notice"] = True
                st.rerun()
    if st.session_state.get("camera_connected_notice", False):
        st.success("Camera configured. The connection is starting in the background.")
        if st.button("Open live wall →", key="camera_to_wall"):
            jump("live", "Live wall")
            st.rerun()
    with st.expander("Camera map and routes"):
        cameras(service,{"id":site},actor)
    @st.fragment(run_every=should_poll(site, m, 4))
    def status():
        st.dataframe(status_rows(service,site,m),hide_index=True)
        activity_map(service,site,m)
    status()


def activity_map(service,site,m):
    from .spatial import load_homography, project_point
    cams=cameras_by_id(service.repo,site)
    open_ids={e["camera_id"] for e in events(service.repo,site) if e["state"] in ("New","Escalated")}
    regions=service.repo.list(site,"coverage_region")
    fig=go.Figure()
    plans = [a for a in service.repo.list(site,"asset") if a.get("role")=="floor_plan"]
    plan = plans[-1] if plans else None
    if plan:
        encoded = base64.b64encode(service.read_asset(plan,"live-map")).decode("ascii")
        fig.add_layout_image(source="data:image/png;base64,"+encoded, xref="x", yref="y",
                             x=0,y=0,sizex=1,sizey=1,sizing="stretch",layer="below")
    calibrations = {c["camera_id"]:c for c in service.floor_calibrations(site)
                    if plan and (c.get("floor_plan_asset") or {}).get("asset_id")==plan["asset_id"]}
    for region in regions:
        x1,y1,x2,y2=region["box"]
        color=tokens()["warn" if region["kind"]=="blind" else "secondary"]
        fig.add_shape(type="rect",x0=x1,y0=y1,x1=x2,y1=y2,line_color=color,fillcolor=color,opacity=.25)
        fig.add_annotation(x=(x1+x2)/2,y=(y1+y2)/2,text=region["label"],showarrow=False)
    for c in cams.values():
        active=c["id"] in open_ids
        sample=m.snapshot(site,c["id"])
        status=sample["status"]
        fig.add_scatter(x=[c.get("x",0)],y=[c.get("y",0)],mode="markers+text",text=[c["label"]],
                        textposition="top center",name=c["label"]+" · "+("Open flag" if active else status),
                        marker=dict(size=16,color=tokens()["critical" if active else "positive" if status=="Online" else "muted"]))
        calibration=calibrations.get(c["id"])
        if calibration and status=="Online" and sample["latest"]:
            H=load_homography(calibration)
            points=[]
            for track in sample["latest"][2]:
                _,x,y,_,height=track["trajectory"][-1]
                try:
                    fx,fy=project_point(H,x,y+height/2)
                except ValueError:
                    continue
                if 0<=fx<=plan["width"] and 0<=fy<=plan["height"]:
                    points.append((fx/plan["width"],fy/plan["height"]))
            if points:
                fig.add_scatter(x=[p[0] for p in points],y=[p[1] for p in points],mode="markers",
                                marker=dict(symbol="cross",color=tokens()["warn"]),
                                name=c["label"]+" · projected sightings")
    if cams or regions:
        fig.update_layout(height=330,xaxis_title="Site map X",yaxis_title="Site map Y",yaxis_autorange="reversed")
        if plan:
            fig.update_xaxes(range=[0,1])
            fig.update_yaxes(range=[1,0],autorange=False)
        plot(fig,key="live_map")
    else:
        st.info("Register camera positions to display the site map.")


def spatial_page(service,site,actor,m):
    from .dashboard import spatial_intelligence
    @st.fragment(run_every=should_poll(site, m, 5))
    def map_panel():
        activity_map(service,site,m)
        st.dataframe(status_rows(service,site,m),hide_index=True)
    map_panel()
    with st.expander("Facility map and calibration"):
        spatial_intelligence(service,{"id":site},actor)


def blind_page(service,site,actor,m):
    from .dashboard import blindsight
    from .services.live.insights import overdue
    if st.button("Open alignment tool in Case mode"):
        jump("case","BlindSight")
        st.rerun()
    with st.expander("Configure visibility, blind regions and portals"):
        blindsight(service,{"id":site},actor)
    cams=cameras_by_id(service.repo,site)
    edges=service.repo.list(site,"transition")
    if edges:
        st.dataframe([{"From":cams[e["from_camera"]]["label"],"To":cams[e["to_camera"]]["label"],
                       "Earliest (s)":e["min_s"],"Latest (s)":e["max_s"]} for e in edges],hide_index=True)
    else:
        st.info("Set directed travel windows in Camera intelligence to evaluate overdue arrivals.")
    @st.fragment(run_every=should_poll(site, m, 8))
    def monitor():
        activity_map(service,site,m)
        leads=overdue(service.repo,m.archive,site,time.time())
        st.subheader("Overdue transit hypotheses")
        if leads:
            st.dataframe(leads,hide_index=True)
        else:
            st.caption("No supported overdue interval in the available camera coverage.")
    monitor()


def live_analytics(service,site,actor,m):
    cams=cameras_by_id(service.repo,site)
    window=st.radio("Occupancy history (minutes)",[15,60],horizontal=True)
    selected=st.selectbox("Heatmap camera",list(cams),format_func=lambda k:cams[k]["label"]) if cams else None
    @st.fragment(run_every=should_poll(site, m, 5))
    def charts():
        current=time.time()
        counts=[]
        for k in cams:
            state=m.snapshot(site,k)
            if state["status"]=="Online" and state["latest"]:
                n=len(state["latest"][2])
                threshold=min((r["threshold"] for r in state["latest"][3] if r["type"]=="crowd"),default=5)
                counts.append({"Camera":cams[k]["label"],"People visible":n,"Crowd":"At threshold" if n>=threshold else "Below threshold","Threshold":threshold})
        st.metric("People visible across online views",sum(r["People visible"] for r in counts) if counts else "—")
        if counts:
            st.dataframe(counts,hide_index=True)
        today=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
        series=m.archive.statistics(site,min(today,current-window*60))
        fig=go.Figure()
        for k,c in cams.items():
            values=[r for r in series if r[0]==k and r[1]*60>=current-window*60]
            if values:
                fig.add_scatter(x=[utc(r[1]*60) for r in values],y=[r[2] for r in values],name=c["label"],mode="lines+markers")
        if fig.data:
            fig.update_layout(yaxis_title="Mean visible people per sample",xaxis_title="UTC")
            plot(fig,key="occupancy_trend")
        log=[e for e in events(service.repo,site) if e["stamp"]>=today]
        hourly=Counter(e["time_utc"][11:13] for e in log)
        if hourly:
            hours=sorted(hourly)
            plot(go.Figure(go.Bar(x=hours,y=[hourly[h] for h in hours])).update_layout(xaxis_title="Hour (UTC)",yaxis_title="Flags"),key="events_hour")
            st.dataframe([{"Rule":r,"Flags today":n} for r,n in Counter(e["rule"] for e in log).most_common()],hide_index=True)
        crossed=Counter(e.get("direction") for e in log if e["method"]=="line")
        st.caption(f"Line crossings today · in (positive axis) {crossed['positive']} · out (negative axis) {crossed['negative']}")
        cells=[np.frombuffer(r[3],dtype=np.int32).reshape(16,16) for r in series if r[0]==selected and r[1]*60>=today]
        if cells:
            fig=go.Figure(go.Heatmap(z=np.sum(cells,axis=0),colorscale=[[0,tokens()["card"]],[1,tokens()["accent"]]]))
            fig.update_layout(title="Today’s sampled positions · camera image",yaxis_autorange="reversed")
            plot(fig,key="live_heat")
    charts()


def live_investigator(service,site,actor,m):
    from .services.live.insights import answer
    from .investigator_engines import PROVIDERS
    minutes=st.selectbox("Search window (minutes)",[5,15,60],index=1)
    planner=None
    with st.expander("Reasoning engine"):
        provider=st.selectbox("Choose engine",["Built-in offline",*PROVIDERS],key="live_engine")
        if provider in PROVIDERS:
            config=PROVIDERS[provider]
            model=st.text_input("Model",config["model"],key="live_model_"+provider)
            key_value=None
            allowed=True
            if config["env"]:
                key_value=st.text_input("Session API key",type="password",key="live_api_"+provider)
                allowed=st.checkbox("Send my question and camera labels to this provider",key="live_consent_"+provider)
            if allowed:
                planner=dict(provider=provider,model=model,api_key=key_value)
            st.caption("The selected engine may choose an event tool. Counts, zone occupancy and citations are calculated locally.")
    key="live_chat_"+site
    history=st.session_state.setdefault(key,[])
    st.caption("Visual match is a lead, not a confirmed identity. Answers use this site’s observations and rule flags.")
    for question,result in history:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            st.write(result["answer"])
            with st.expander("Evidence used"):
                st.caption(result.get("engine","Local live tools")+" · "+result["tool"]+" · "+result["window"])
                if result["rows"]:
                    st.dataframe(result["rows"],hide_index=True)
    if q:=st.chat_input("Ask about alerts, line crossings or zone occupancy"):
        result=answer(service.repo,m,site,q,minutes,planner=planner)
        history.append((q,result))
        st.rerun()


def health(service,site,actor,m):
    cfg=configs(service.repo,site)
    cams=cameras_by_id(service.repo,site)
    @st.fragment(run_every=should_poll(site, m, 3))
    def health_table():
        if m.retention_error:
            st.error(m.retention_error)
        st.dataframe(status_rows(service,site,m),hide_index=True)
        st.caption(f"Person detector · {'YOLO11n loaded' if getattr(m.detector,'model',None) else 'Waiting for first inference'}  |  Tracker · Motion IoU  |  Embeddings · case indexing only  |  Investigator · local tools")
        for k in cfg:
            for ts,msg in m.snapshot(site,k)["errors"][-3:]:
                st.caption(f"{ts} · {cams[k]['label']} · {msg}")
    health_table()
    if cfg:
        k=st.selectbox("Stream",list(cfg),format_func=lambda k:cams[k]["label"])
        a,b=st.columns(2)
        if a.button("Start / restart stream"):
            m.restart(site,cfg[k])
            service.repo.audit(site,actor,"stream.restart",k)
            st.rerun()
        if b.button("Stop stream"):
            m.stop(site,k)
            service.repo.audit(site,actor,"stream.stop",k)
            st.rerun()
    settings=service.repo.list(site,"retention")
    last=settings[-1] if settings else {}
    with st.form("retention_"+site):
        mins=st.number_input("Keep recent footage (minutes)",1,120,last.get("minutes",15))
        gb=st.number_input("Rolling footage budget (GiB)",.02,8.,float(last.get("gb",.25)),step=.05)
        if st.form_submit_button("Apply retention"):
            service.repo.put("retention",dict(case_id=site,minutes=mins,gb=gb),actor)
            st.success("Settings saved. The buffer is pruned every ten seconds.")

