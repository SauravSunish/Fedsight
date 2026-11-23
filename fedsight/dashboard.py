"""Presentation layer for the local FedSight evidence workbench."""
from __future__ import annotations
from html import escape
import dataclasses
import io
import json
import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from .core import RejectionPolicy, TimeAnchor, canonical, digest
from .service import FedSight, environment_versions
from .repository import CASE_TRANSITIONS
from .theme import apply_theme, palette, tokens, translucent, plot
from .cctv_matrix import (
    get_active_tracks_at_time,
    extract_or_render_camera_frame,
    compose_cctv_matrix,
)
from .dossier import (
    render_biometric_badges,
    render_velocity_sparkline,
    render_custody_stamp,
    render_tactical_dossier_header,
)

from .navigation import DEFAULT, REGISTRY, pages, resolve, workspace_menu

PAGES = pages("case")


@st.cache_data(max_entries=32, show_spinner=False)
def _quick_jump_tracks(root, case_id, revision, _service):
    return _service.active_tracks(case_id)[:10]


def main():
    st.set_page_config(page_title="FedSight · Intelligence Workbench", page_icon="◈", layout="wide")
    root = Path(os.environ.get("FEDSIGHT_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "workbench"))
    service = FedSight(root)
    repo = service.repo
    from . import operations as ops
    state = st.session_state
    history = state.setdefault("last_page", dict(DEFAULT))
    previous = state.get("mode")
    url = (st.query_params.get("mode"), st.query_params.get("page"))
    requested = state.pop("route_request", None)
    if requested or previous is None or url != state.get("last_url"):
        mode, page = resolve(*(requested or url), history)
    else:
        mode, page = resolve(previous, state.get("workspace"), history)
    if previous and previous != mode:
        repo.audit(state.get("case_memory") or "workstation", state.get("reviewer") or "local-reviewer",
                   "mode_switch", f"{previous}/{mode}")
    state.mode = mode
    state.workspace = page
    state.mode_switch = mode
    history[mode] = page
    current_url = (mode, REGISTRY[page].slug)
    if current_url != url:
        st.query_params.update(mode=mode, page=current_url[1])
    state.last_url = current_url
    apply_theme(mode)

    def switch_mode():
        state.route_request = (state.mode_switch or state.mode, None)

    def on_mode_toggle():
        new_mode = "live" if state.get("mode_toggle") else "case"
        if new_mode == "case":
            try:
                from .operations import manager
                m = manager(str(service.root))
                for s in repo.list(None, "live_site"):
                    m.pause_all(s["id"])
            except Exception:
                pass
        state.route_request = (new_mode, None)
        state.mode_switch = new_mode

    with st.sidebar:
        st.markdown('<div class="brand"><span>◈</span> FEDSIGHT</div><div class="brand-sub">INTELLIGENCE WORKBENCH</div>', unsafe_allow_html=True)
        
        is_live = (mode == "live")
        if state.get("mode_toggle") != is_live:
            state.mode_toggle = is_live

        case_active = "active" if not is_live else ""
        live_active = "active" if is_live else ""

        m_c1, m_c2, m_c3 = st.columns([1, 1.6, 1.1], vertical_alignment="center")
        with m_c1:
            st.markdown(f'<div class="mode-badge-label case {case_active}" style="justify-content:flex-end;">CASE</div>', unsafe_allow_html=True)
        with m_c2:
            st.toggle(
                "Workbench Mode",
                value=is_live,
                key="mode_toggle",
                on_change=on_mode_toggle,
                label_visibility="collapsed",
            )
        with m_c3:
            st.markdown(f'<div class="mode-badge-label live {live_active}"><span class="pulse-beacon-dot"></span>LIVE</div>', unsafe_allow_html=True)

        actor = st.text_input("Reviewer name", "local-reviewer", key="reviewer", help="Attribution on this local workstation; not an authenticated account.").strip()
        cases = {c["id"]: c for c in repo.list(None, "case")}
        # Selection requests are consumed before creating the widget on the next rerun.
        requested = st.session_state.pop("open_case", None)
        if requested in cases:
            st.session_state.selected_case = requested
        if st.session_state.get("selected_case") not in cases:
            st.session_state.selected_case = state.case_memory if state.get("case_memory") in cases else (list(cases)[-1] if cases else None)
        if mode == "live":
            site = ops.site_picker(service, actor or "local-reviewer")
            cid = None
        else:
            if not requested and state.get("case_memory") in cases and "selected_case" not in state:
                state.selected_case = state.case_memory
            cid = st.selectbox("Active case", list(cases), key="selected_case", format_func=lambda c: f"{cases[c]['title']} · {c[-8:]}") if cases else None
            state.case_memory = cid
            st.button("Manage cases", on_click=lambda: state.update(workspace="Cases"))
        if cid:
            with repo.connect() as conn:
                revision = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM audit WHERE case_id=?", (cid,)).fetchone()[0]
            tracks = _quick_jump_tracks(str(root.resolve()), cid, revision, service)
        else:
            tracks = []
        cams = repo.list(cid, "camera") if cid else []
        destinations = {p: ("workspace", p) for p in pages(mode)}
        for camera in cams:
            destinations[f"Camera · {camera['label']} · {camera['id'][-8:]}"] = ("camera", camera["id"])
        for track in tracks[:10]:
            destinations[f"Sighting · {track['id'][-8:]} · {track['start_time']:.1f}s"] = ("track", track)

        def quick_jump():
            choice = st.session_state.global_cmd_palette
            if choice not in destinations:
                return
            kind, target = destinations[choice]
            if kind == "camera":
                st.session_state.workspace = "Camera intelligence"
                st.session_state["camera_focus_" + cid] = target
            elif kind == "track":
                st.session_state.workspace = "Video review"
                st.session_state.investigator_jump = dict(case_id=cid, source_id=target["source_id"],
                                                         seconds=target["start_time"], track_id=target["id"])
            else:
                st.session_state.workspace = target
            st.session_state.global_cmd_palette = None

        st.selectbox("Quick jump", list(destinations), index=None, placeholder="Search destinations",
                     help="Find a workspace, a case camera or one of the first ten indexed sightings.",
                     key="global_cmd_palette", on_change=quick_jump)
        with st.expander("Workspace shortcuts"):
            for route in REGISTRY.values():
                case_tool = mode == "case" and len(route.modes) > 1
                if mode in route.modes and (route.shortcut or case_tool):
                    st.button(route.shortcut or route.title, use_container_width=True,
                              on_click=lambda target=route.title: state.update(workspace=target))
        page = st.radio("Workspaces", workspace_menu(mode,page), key="workspace")
        history[mode] = page
        beginner_help = st.checkbox("Show beginner guidance", value=True, key="beginner_guidance",
                                     help="Show plain-language tips and score explanations. You can turn this off once you know the workflow.")
        st.divider()
        st.caption("Local workstation · Human-reviewed proposals")
        st.caption("Consented footage only. Appearance links do not establish real-world identity.")
    breadcrumb = "LIVE OPERATIONS" if mode == "live" else "INVESTIGATION WORKSPACE"
    st.markdown(f'<div class="eyebrow">FEDSIGHT / {breadcrumb}</div>', unsafe_allow_html=True)
    st.title(page)
    case = cases.get(cid)
    if mode == "live":
        if not actor:
            st.warning("Enter a reviewer name to continue.")
            return
        ops.live_page(service, site, actor, page, beginner_help)
        return
    if case:
        st.markdown(
            f'<div class="tactical-ribbon"><span>Case <strong>{escape(case["title"])}</strong></span>'
            f'<span class="tactical-chip-cyan">{escape(case["status"])}</span>'
            f'<span>Reviewer · {escape(actor)}</span></div>', unsafe_allow_html=True)
    if beginner_help:
        from .workspace_guides import show_workspace_guide
        show_workspace_guide(st, page)
    if case:
        if case["demo"]:
            st.info("SYNTHETIC DEMO — illustrated tracklets and generated vectors. These are not measured Re-ID results.")
    if not actor:
        st.warning("Enter a reviewer name to continue.")
        st.stop()
    try:
        if case or page in {"Overview", "Cases"}:
            globals()[REGISTRY[page].handler](service, case, actor)
        else:
            st.info("Create a case or load the synthetic walkthrough from Overview.")
    except (ValueError, OSError, KeyError, sqlite3.OperationalError) as exc:
        st.error(str(exc))


def create_case_form(service, actor):
    with st.form("new_case", clear_on_submit=True):
        title = st.text_input("Case title")
        description = st.text_area("Description / research question")
        consent = st.text_area("Collection consent and retention notes")
        if st.form_submit_button("Create case", type="primary"):
            record = service.repo.create_case(title, description, actor, consent=consent)
            st.session_state.open_case = record["id"]
            st.rerun()


def overview(service, case, actor):
    if case:
        repo, cid = service.repo, case["id"]
        tracks, proposals, reviews = service.active_tracks(cid), repo.active_proposals(cid), repo.latest_reviews(cid)
        sources = repo.list(cid, "source")
        cameras = repo.list(cid, "camera")
        jobs = repo.list(cid, "job")
        latest_jobs = {job["source_id"]: job for job in jobs}
        pending = [source for source in sources if latest_jobs.get(source["id"], {}).get("status") != "completed"]
        st.markdown(f'<div class="overview-hero"><div class="overview-kicker">ACTIVE INVESTIGATION</div>'
                    f'<div class="overview-title">{escape(case["title"])}</div>'
                    f'<div class="overview-copy">{len(sources)} recordings · {len(cameras)} cameras · '
                    f'{len(tracks)} indexed sightings</div></div>', unsafe_allow_html=True)
        if not sources:
            next_step, destination = "Add your first recording", "Evidence & processing"
        elif pending and not case.get("demo"):
            next_step, destination = "Process recordings", "Evidence & processing"
        elif tracks:
            next_step, destination = "Review video evidence", "Video review"
        else:
            next_step, destination = "Check processing", "Evidence & processing"
        st.button(next_step + " →", type="primary", key="overview_next",
                  on_click=lambda: st.session_state.update(workspace=destination))
        events = repo.list(cid, "event")
        if sources:
            with st.expander("Recordings and camera coverage", expanded=True):
                names = {camera["id"]: camera["label"] for camera in cameras}
                st.dataframe([{"Recording": s["name"], "Camera": names.get(s.get("camera_id"), "Unassigned"),
                               "Processing": latest_jobs.get(s["id"], {}).get("status", "Not processed")}
                              for s in sources], hide_index=True, use_container_width=True)
                st.button("Open spatial intelligence", key="overview_spatial",
                          on_click=lambda: st.session_state.update(workspace="Spatial intelligence"))
        today = pd.Timestamp.now(tz="UTC").date().isoformat()
        for col, label, value in zip(st.columns(4), ["EVIDENCE SOURCES", "INDEXED TRACKLETS", "EVENTS RECORDED TODAY", "AWAITING REVIEW"],
                                     [len(repo.list(cid, "source")), len(tracks), sum(e.get("created_at", "")[:10] == today for e in events), sum(reviews.get(p["id"], {}).get("decision") not in {"accept", "reject"} for p in proposals)]):
            col.metric(label, value)
        st.button("Ask Investigator →", on_click=lambda: st.session_state.update(workspace="FedSight Investigator"))
        intrusions = [e for e in events if e.get("method") == "zone"][-4:][::-1]
        if intrusions:
            with st.expander("Recent zone events", expanded=True):
                by_track = {t["id"]: t for t in tracks}
                for event in intrusions:
                    left, right = st.columns([1, 5])
                    track = by_track.get(event.get("track_id"))
                    if track and track.get("preview_asset"):
                        left.image(service.read_asset(track["preview_asset"], actor), width=90)
                    right.write(f"Zone flag · clip {event.get('start_time', 0):.1f}s · source {event['source_id'][-8:]}")
                    right.caption("Batch event · review original footage")
    from .operations import recent_events
    recent_events(service, actor)
    st.subheader("Your intelligence workspace")
    for column, title, description, destination in zip(st.columns(3), ["Review", "Respond", "Research"],
                                          ["Search recordings, compare sightings and review evidence.",
                                           "Configure events and monitor a local camera.",
                                           "Explore movement, activity and patterns."],
                                          ["Search & digest", "Rules & events", "Analytics"]):
        with column, st.container(border=True):
            st.markdown(f"### {title}")
            st.write(description)
            st.button("Open " + destination, key="overview_" + destination, use_container_width=True,
                      on_click=lambda target=destination: st.session_state.update(workspace=target))
    left, right = st.columns([1.5, 1])
    with left, st.expander("Create a new case", expanded=case is None):
        create_case_form(service, actor)
    with right, st.container(border=True):
        st.subheader("Explore the full workflow")
        st.write("Three virtual cameras. Ten illustrated tracklets. Reviewable proposals, route explanations and batch alerts.")
        if st.button("Load synthetic walkthrough", type="primary", use_container_width=True):
            from .demo import create_demo
            st.session_state.open_case = create_demo(service, actor)
            st.rerun()
    if not case:
        return
    st.subheader("Case record")
    st.write(case["description"] or "No description recorded.")
    st.caption("Collection / retention: " + (case.get("consent") or "No notes recorded."))
    states = sorted(CASE_TRANSITIONS[case["status"]] - {"Processing"})
    if states:
        state = st.selectbox("Change case status", states)
        if st.button("Record status change"):
            if state == "Closed" and not all(a["ok"] for a in service.verify_assets(cid, actor)):
                raise ValueError("Asset verification must pass before closure.")
            repo.transition(cid, state, actor)
            st.rerun()
    with st.form("case_note"):
        note = st.text_area("Add a case note")
        if st.form_submit_button("Save note"):
            if not note.strip():
                raise ValueError("Enter a note first.")
            repo.put("note", {"case_id": cid, "text": note.strip(), "actor": actor}, actor)
            st.rerun()
    for note in repo.list(cid, "note")[-5:][::-1]:
        st.markdown(f"**{note['actor']}** · {note['created_at'][:19]}")
        st.write(note["text"])


def evidence(service, case, actor):
    from .processing_queue import QUEUE
    repo, cid = service.repo, case["id"]
    sources = repo.list(cid, "source")
    mapping = {c["id"]: c["label"] for c in repo.list(cid, "camera")}
    if not case["demo"]:
        st.subheader("Register original evidence")
        uploads = st.file_uploader("Videos · up to 200 MB each", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)
        camera = st.selectbox("Source camera for this batch", [None] + list(mapping), format_func=lambda c: mapping.get(c, "Unknown"))
        with st.expander("Clock alignment — optional"):
            original = st.text_input("Recording start with UTC offset", placeholder="2026-11-11T10:30:00+05:30")
            tz = st.text_input("Original timezone label", "unknown")
            cols = st.columns(3)
            offset = cols[0].number_input("Correction (seconds)", value=0.)
            uncertainty = cols[1].number_input("Uncertainty ± seconds", min_value=0., value=0.)
            drift = cols[2].number_input("Clock drift (ppm)", value=0.)
            method = st.text_input("Alignment method / provenance")
            st.caption("Leave the timestamp empty when unknown. Upload anchored recordings individually.")
        if st.button("Preserve and register originals", disabled=not uploads, type="primary"):
            if original and len(uploads) > 1:
                raise ValueError("Upload anchored recordings individually so each has its own start time.")
            anchor = dataclasses.asdict(TimeAnchor(original or None, tz, offset, uncertainty, drift, method or "unknown"))
            for upload in uploads:
                service.ingest(cid, upload.name, upload.getvalue(), actor, camera_id=camera, time_anchor=anchor)
            st.rerun()
    if sources:
        st.dataframe([{"Source": s["name"], "Source ID": s["id"], "Camera": mapping.get(s.get("camera_id"), "Unknown"),
                       "Original SHA-256": s.get("sha256") or "Synthetic", "Clock": s.get("time_anchor", {}).get("original_timestamp") or "Relative only"} for s in sources], hide_index=True, use_container_width=True)
    jobs = repo.list(cid, "job")
    if jobs:
        st.subheader("Processing history")
        st.dataframe([{"Source": j["source_id"], "Status": j["status"], "Frames": j.get("meta", {}).get("processed_frames"),
                       "Incomplete": j.get("meta", {}).get("incomplete"), "Frame step used": j.get("meta", {}).get("effective_frame_step"),
                       "Tracker used": j.get("meta", {}).get("actual_tracker", "synthetic" if j.get("demo") else None),
                       "Note": j.get("error", j.get("embedding_note", ""))} for j in jobs], hide_index=True, use_container_width=True)
        latest = {j["source_id"]: j for j in jobs}
        for source in sources:
            failure = latest.get(source["id"], {})
            if failure.get("status") == "failed":
                st.error(f"{source['name']}: {failure['error']}")
    if case["demo"]:
        st.caption("Demo sources have no original video. Create a real case to process recordings.")
        return
    st.subheader("Perception pipeline")
    c1, c2, c3 = st.columns(3)
    detector = c1.selectbox("Detector", ["yolo11n.pt", "yolo26n.pt", "yolov8n.pt"])
    tracker = c2.selectbox("Tracker", ["auto", "bytetrack.yaml", "botsort.yaml", "portable-iou"],
                           help="Auto prefers ByteTrack and selects portable motion/IoU when native LAP is unavailable. The actual tracker is recorded.")
    confidence = c3.slider("Detector threshold", .1, .9, .25, .05)
    step = c1.slider("Frame sampling step", 1, 5, 2)
    frames = c2.slider("Maximum sampled frames", 100, 2000, 800, 100)
    samples = c3.slider("Crops per track", 2, 24, 8)
    imgsz = st.selectbox("Detector input resolution", [320, 480, 640, 960, 1280], index=2,
                         help="YOLO resizes the image for inference; original evidence and saved coordinates retain the source resolution.")
    categories = st.multiselect("Detect and index", ["Person", "Vehicle", "Bag"], default=["Person"])
    st.caption("Frame spacing increases automatically when needed to sample the whole recording within the frame budget. Clips up to 5 minutes are accepted; processing may take several minutes on CPU.")
    weights = st.text_input("Trusted local OSNet Re-ID weights (optional)")
    architecture = st.selectbox("OSNet architecture", ["osnet_x1_0", "osnet_ain_x1_0", "osnet_x0_25"])
    if not weights:
        st.warning("No person Re-ID checkpoint selected. Default ImageNet weights are a feature baseline, not a trained person Re-ID model.")
    scope = st.radio("Processing scope", ["Pending / failed sources", "Reprocess every source with these settings"], horizontal=True)
    task = QUEUE.get(str(service.root.resolve()), cid)
    running = bool(task and task.snapshot()["status"] in {"queued", "running", "cancelling"})
    if st.button("Process videos", type="primary", disabled=running or not sources or case["status"] not in {"Created", "Under Review", "Failed", "Reviewed"}):
        from .video import AnalyzeConfig
        from .embeddings import OSNetConfig
        if not categories:
            raise ValueError("Select at least one object category.")
        classes = tuple(([0] if "Person" in categories else []) + ([1, 2, 3, 5, 7] if "Vehicle" in categories else []) + ([24, 26, 28] if "Bag" in categories else []))
        QUEUE.start(service, cid, actor, AnalyzeConfig(weights=detector, tracker=tracker, confidence=confidence, classes=classes, imgsz=imgsz, frame_step=step, max_processed_frames=frames, max_crops=samples),
                    OSNetConfig(model_name=architecture, model_path=weights.strip() or None), force=scope.startswith("Reprocess"))
        st.session_state["processing_observed_" + cid] = True
        st.rerun()
    @st.fragment(run_every=2 if running else None)
    def processing_status():
        current = QUEUE.get(str(service.root.resolve()), cid)
        if not current:
            return
        state = current.snapshot()
        st.progress(state["progress"], text=f"{state['stage']} · {state['source']}".rstrip(" ·"))
        if state["status"] in {"queued", "running", "cancelling"}:
            if state.get("eta_s") is not None:
                st.caption(f"Rough sampling estimate: {state['eta_s'] / 60:.1f} minutes remaining; embedding and ranking can take longer.")
            if st.button("Cancel processing", disabled=state["status"] == "cancelling"):
                QUEUE.cancel(str(service.root.resolve()), cid)
                st.rerun(scope="fragment")
            st.caption("Cancellation takes effect between frames or after the current model operation. Completed recordings are retained.")
        else:
            if state["error"]:
                st.error(state["error"])
            elif state["status"] == "cancelled":
                st.warning("Processing cancelled. Select pending / failed sources to retry unfinished recordings.")
            elif state["status"] == "completed":
                st.success("Processing completed.")
                st.button("Review processed video →", key="processing_to_review",
                          on_click=lambda: st.session_state.update(workspace="Video review"))
                st.button("Ask Investigator →", key="processing_to_investigator",
                          on_click=lambda: st.session_state.update(workspace="FedSight Investigator"))
            if st.session_state.pop("processing_observed_" + cid, False):
                st.rerun()
    processing_status()
    if case["status"] == "Processing" and not running:
        st.warning("A processing run is active or was interrupted. Recover only after the original worker has stopped.")
        if st.button("Mark interrupted processing as failed"):
            repo.transition(cid, "Failed", actor)
            st.rerun()
    with st.expander("Runtime versions"):
        st.json(environment_versions())


def show_track(service, track, source, actor, key, playback=False):
    st.markdown(render_tactical_dossier_header(track, source), unsafe_allow_html=True)
    if track.get("preview_asset"):
        st.image(Image.open(io.BytesIO(service.read_asset(track["preview_asset"], actor))), width=210)
    st.markdown(f"**{source['name']}**")
    st.caption(f"Clip time · {track['start_time']:.2f}–{track['end_time']:.2f} s")
    anchor_dict = source.get("time_anchor")
    if anchor_dict:
        try:
            anchor = TimeAnchor(**anchor_dict)
            bounds = anchor.interval(track["start_time"])
            if bounds:
                instant = pd.to_datetime(sum(bounds) / 2, unit="s", utc=True)
                st.caption(f"UTC {instant.strftime('%Y-%m-%d %H:%M:%S')} ±{anchor.uncertainty_s:g}s · SHA-256 {(source.get('sha256') or 'synthetic')[:8]}")
            else:
                st.caption(f"Clock unaligned · SHA-256 {(source.get('sha256') or 'synthetic')[:8]}")
        except Exception:
            st.caption(f"SHA-256 {(source.get('sha256') or 'synthetic')[:8]}")
    else:
        st.caption(f"SHA-256 {(source.get('sha256') or 'synthetic')[:8]}")
    from .operations import watch_button
    watch_button(service, track, actor, key)
    if playback and source.get("asset") and st.checkbox("Load original context video", key=f"play_{key}"):
        st.video(service.read_asset(source["asset"], actor), start_time=max(0., track["start_time"] - 2), end_time=track["end_time"] + 2)
        st.caption("Original recording with bounded playback. AVI/MKV support depends on browser codecs.")


def render_wall_of_eyes(service, case, actor, sources: dict, tracks: list[dict], available: list[str]):
    """Compare source frames at a shared clip-relative offset."""
    st.caption("Compare up to four recordings at the same elapsed time from each video's start. Cameras may have started recording at different times.")
    selected = st.multiselect("Recordings on the wall", available, default=available[:4], max_selections=4,
                              format_func=lambda sid: sources[sid]["name"], key="wall_sources_" + case["id"])
    if not selected:
        st.info("Select a recording to see its frames here.")
        return
    jobs = {j["id"]: j for j in service.repo.list(case["id"], "job")}
    max_offset = 0.
    for track in tracks:
        if track["source_id"] in selected:
            meta = jobs.get(track.get("job_id"), {}).get("meta", {})
            fps = meta.get("fps") or 25.
            last_frame = max(0, (meta.get("total_frames") or 0) - 1)
            max_offset = max(max_offset, track.get("end_time", 0.), last_frame / fps)
    moment = st.slider("Time from start of each recording", 0., max(.1, float(max_offset)), 0.,
                       step=.1, format="%.1f s", key="wall_offset_" + case["id"])
    camera_labels = {c["id"]: c["label"] for c in service.repo.list(case["id"], "camera")}
    alignments = service.blind_alignments(case["id"])
    tiles = []
    for index, sid in enumerate(selected):
        if index % 2 == 0:
            columns = st.columns(min(2, len(selected)))
        src = sources[sid]
        if sid in alignments:
            alignment = alignments[sid]
            src = {**src, "camera_id": alignment["camera_id"], "time_anchor": alignment["time_anchor"]}
        label = camera_labels.get(src.get("camera_id"), src["name"])
        active = get_active_tracks_at_time(tracks, sid, moment)
        with columns[index % 2], st.container(border=True):
            st.write(src["name"])
            try:
                frame = extract_or_render_camera_frame(service, src, label, active, moment, actor)
            except (OSError, ValueError) as exc:
                st.warning(f"Frame unavailable · {exc}")
                continue
            st.image(frame, use_container_width=True)
            st.caption(f"{len(active)} indexed sightings at {moment:.1f} s")
            if active:
                with st.expander(f"Sighting details ({len(active)})"):
                    for track in active:
                        st.write(f"{track.get('object_class', 'person').title()} · {track['id'][-8:]} · "
                                 f"{track['start_time']:.1f}–{track['end_time']:.1f} s")
                        st.markdown(render_biometric_badges(track), unsafe_allow_html=True)
                        fps = jobs.get(track.get("job_id"), {}).get("meta", {}).get("fps")
                        st.markdown(render_velocity_sparkline(track.get("trajectory"), fps), unsafe_allow_html=True)
            tiles.append((sid, frame))
    if tiles and len(tiles) == len(selected):
        mosaic = compose_cctv_matrix(tiles, grid=(min(2, len(tiles)), (len(tiles) + 1) // 2))
        buf = io.BytesIO()
        mosaic.save(buf, format="PNG")
        st.download_button("Download camera wall snapshot", buf.getvalue(),
                           f"camera_wall_{moment:.1f}s.png", "image/png")


def video_review(service, case, actor):
    """Review indexed source footage and inspect sampled tracklet overlays."""
    import tempfile
    import cv2
    from .review_video import annotate_frame, track_density
    from .render_queue import QUEUE as RENDER_QUEUE
    repo, cid = service.repo, case["id"]
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    tracks = service.active_tracks(cid)
    available = [sid for sid in sources if any(t["source_id"] == sid for t in tracks)]
    for sid, src in sources.items():
        if sid not in available and src.get("provenance", {}).get("source_type") == "live":
            st.video(service.read_asset(src["asset"], actor), start_time=int(src["event_seconds"]))
            st.caption(f"{src['name']} · rule flag at {src['event_seconds']:.2f}s")
    if not available:
        st.info("Process a recording or load the synthetic walkthrough to review tracklets.")
        return

    jump = st.session_state.pop("investigator_jump", None)
    if jump and (jump.get("case_id") != cid or jump.get("source_id") not in available):
        jump = None
    if jump:
        st.session_state.video_review_mode = "Single recording"
        st.session_state.video_review_source = jump["source_id"]
        st.session_state.update(vr_category="All", vr_colour="All", vr_duration=0.)
    vr_mode = st.radio("Review mode", ["Camera wall", "Single recording"], horizontal=True, key="video_review_mode")
    if vr_mode == "Camera wall":
        render_wall_of_eyes(service, case, actor, sources, tracks, available)
        return
    if st.session_state.get("video_review_source") not in available:
        st.session_state["video_review_source"] = available[0]
    sid = st.selectbox("Recording", available, key="video_review_source", format_func=lambda x: sources[x]["name"])
    source = sources[sid]
    provenance = source.get("provenance", {})
    if provenance.get("source_type") == "live":
        captured = "Simulated replay" if provenance["simulated"] else "Camera capture"
        st.caption(f"{captured} · capture UTC {provenance['event_utc']} · live event {provenance['live_event_id'][-8:]}")
    selected_tracks = [t for t in tracks if t["source_id"] == sid]
    events = [e for e in repo.list(cid, "event") if e["source_id"] == sid]
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Object class", ["All", "person", "vehicle", "bag"], key="vr_category")
        colours = sorted({t.get("attributes", {}).get("color", "unknown") for t in selected_tracks})
        colour = c2.selectbox("Approximate colour", ["All"] + colours, key="vr_colour")
        minimum = c3.number_input("Minimum visible duration (s)", min_value=0., value=0., key="vr_duration")
    filtered = [t for t in selected_tracks if (category == "All" or t.get("object_class", "person") == category)
                and (colour == "All" or t.get("attributes", {}).get("color", "unknown") == colour)
                and t.get("duration_s", 0) >= minimum]
    if jump and jump.get("track_id"):
        jump_match = next((t for t in filtered if t["id"] == jump["track_id"]), None)
        if jump_match:
            st.session_state["vr_jump_" + sid] = jump_match
    st.caption(f"{len(filtered)} tracklets · sampled detections can miss or fragment objects. Confirm all leads in the original recording.")
    job_map = {j["id"]: j for j in repo.list(cid, "job")}
    job = job_map.get(selected_tracks[0].get("job_id"), {})
    meta = job.get("meta", {})
    fps = float(meta.get("fps") or 25.)
    alignments = service.blind_alignments(cid)
    time_anchor = alignments.get(sid, source).get("time_anchor", {})
    camera_id = alignments.get(sid, source).get("camera_id")
    states = {s["rule_id"]: s for s in repo.list(cid, "rule_state")}
    overlay_rules = [r for r in repo.list(cid, "rule") if r.get("type") in {"zone", "line"}
                     and states.get(r["id"], {}).get("enabled", True)
                     and (not r.get("camera_id") or r["camera_id"] == camera_id)]
    latest_reviews = repo.latest_reviews(cid)
    validated_ids = set()
    for proposal in repo.active_proposals(cid):
        if latest_reviews.get(proposal["id"], {}).get("decision") == "accept":
            validated_ids.update([proposal["left_id"], proposal["right_id"]])
    with st.expander("Overlay controls", expanded=False):
        o1, o2, o3 = st.columns(3)
        overlay_options = {"boxes": o1.checkbox("Boxes", value=True), "ids": o1.checkbox("Track IDs", value=True),
                           "trails": o2.checkbox("Motion trails", value=True), "zones": o2.checkbox("Zones and lines", value=True),
                           "hud": o3.checkbox("HUD cards", value=True), "timestamp": o3.checkbox("Timestamp", value=True)}
        overlay_options["crowd_threshold"] = st.number_input("Crowd alert threshold", min_value=1, max_value=64, value=5)
    total = float(meta.get("total_frames") or 0)
    duration = max(total / fps if fps else 0, max((t.get("end_time", 0) for t in selected_tracks), default=0), .01)
    density = track_density(filtered, duration)
    fig = go.Figure(go.Bar(x=[d["clip_seconds"] for d in density], y=[d["tracklets"] for d in density], marker_color=palette()[0], name="Tracklets"))
    if events:
        fig.add_trace(go.Scatter(x=[e["start_time"] for e in events], y=[0] * len(events), mode="markers", marker=dict(symbol="triangle-up", size=11, color=palette()[1]), name="Rule events", text=[e["method"] for e in events]))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=10, b=25), xaxis_title="Clip time (seconds)", yaxis_title="Tracklets", legend=dict(orientation="h"))
    plot(fig, use_container_width=True)
    if filtered:
        chosen_track = st.selectbox("Jump to a tracklet", filtered,
                                    format_func=lambda t: f"{t['id'][-8:]} · {t.get('object_class', 'person')} · {t['start_time']:.1f}s",
                                    key="vr_jump_" + sid)
        if st.button("Jump to tracklet"):
            st.session_state["vr_frame_" + sid] = min(max(0, int(duration * fps) - 1), max(0, int(chosen_track["start_time"] * fps)))
        overlay_options["selected_ids"] = [chosen_track["id"]]
        overlay_options["highlight"] = st.checkbox("Highlight selected tracklet")
    if filtered:
        from .operations import watch_button
        watch_button(service, chosen_track, actor, "video_review")
    if events:
        selected_event = st.selectbox("Jump to event", events, format_func=lambda e: f"{e['method']} · {e['start_time']:.1f}s · {e['id'][-6:]}")
        if st.button("Jump to event"):
            st.session_state["vr_frame_" + sid] = min(max(0, int(duration * fps) - 1), max(0, int(selected_event["start_time"] * fps)))
    max_frame = max(1, int(total or duration * fps) - 1)
    if jump and jump.get("case_id") == cid and jump.get("source_id") == sid:
        st.session_state["vr_frame_" + sid] = min(max_frame, max(0, int(jump["seconds"] * fps)))
    review_frame = st.slider("Review frame", 0, max_frame, 0, key="vr_frame_" + sid)
    moment = review_frame / fps
    st.caption(f"Clip time {moment:.2f}s · frame {review_frame} at {fps:g} fps")
    if source.get("asset"):
        blob = service.read_asset(source["asset"], actor)
        st.video(blob, start_time=int(moment))
        with st.spinner("Opening the selected source frame…"):
            path = None
            try:
                with tempfile.NamedTemporaryFile(dir=service.work, suffix=source.get("suffix", ".mp4"), delete=False) as tmp:
                    tmp.write(blob)
                    path = Path(tmp.name)
                os.chmod(path, 0o600)
                cap = cv2.VideoCapture(str(path))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, review_frame)
                    ok, frame = cap.read()
                    frame_no = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                finally:
                    cap.release()
                if ok:
                    annotated = annotate_frame(frame, filtered, frame_no, options=overlay_options, fps=fps,
                                               time_anchor=time_anchor, validated_ids=validated_ids, rules=overlay_rules)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"Annotated source frame {frame_no} · {moment:.2f}s. Boxes between nearby sampled points are interpolated.", use_container_width=True)
                    png = cv2.imencode(".png", annotated)[1].tobytes()
                    if st.download_button("Download annotated snapshot", png, f"fedsight-frame-{frame_no}.png", "image/png"):
                        from .core import digest
                        repo.audit(cid, actor, "review.snapshot_downloaded", sid, digest(png))
                else:
                    st.warning("The selected frame could not be decoded. Try another moment.")
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
        with st.expander("Create an annotated review clip"):
            st.caption("Creates a silent MP4 in the background. Boxes use sampled detections and may disappear between distant samples.")
            export_scope = st.radio("Video to render", ["Entire video", "Clip from selected frame"],
                                    horizontal=True, key="vr_export_scope_" + sid)
            if export_scope == "Entire video":
                render_start, length = 0., min(duration, 300.)
                st.caption(f"Renders from the beginning for up to {length:.1f} seconds. Longer videos take more time and disk space.")
            else:
                remaining = max(0., min(300., duration - moment))
                length_key = "vr_export_length_" + sid
                if length_key in st.session_state:
                    st.session_state[length_key] = min(max(0.1, remaining), max(0.1, st.session_state[length_key]))
                length = st.number_input("Clip duration (seconds)", min_value=0.1,
                                         max_value=max(0.1, remaining), value=max(0.1, min(10., remaining)),
                                         key=length_key)
                render_start = moment
            root = str(service.root.resolve())
            render_task = RENDER_QUEUE.get(root, cid, sid)
            rendering = bool(render_task and render_task.snapshot()["status"] in {"queued", "running"})
            if st.button("Render annotated clip", key="vr_render", disabled=rendering):
                RENDER_QUEUE.start(service, cid, source, actor, filtered, start_s=render_start, duration_s=float(length),
                                   options=overlay_options, time_anchor=time_anchor, validated_ids=validated_ids, rules=overlay_rules)
                st.session_state["vr_render_observed_" + sid] = True
                st.rerun()
            @st.fragment(run_every=2 if rendering else None)
            def show_render():
                task = RENDER_QUEUE.get(root, cid, sid)
                if not task:
                    return
                state = task.snapshot()
                if state["status"] in {"completed", "failed"} and st.session_state.pop("vr_render_observed_" + sid, False):
                    st.rerun()
                if state["status"] in {"queued", "running"}:
                    st.info("Rendering and encoding in the background…")
                elif state["status"] == "failed":
                    st.error(state["error"] or "Rendering failed.")
                elif state["status"] == "completed":
                    asset = state["result"]["asset"]
                    rendered = service.read_asset(asset, actor)
                    st.video(rendered)
                    st.download_button("Download annotated MP4", rendered, f"fedsight-annotated-{sid[-8:]}.mp4", "video/mp4")
                    st.caption("Derived clip SHA-256: " + asset["sha256"] + " · Codec: " + state["result"]["render_meta"]["codec"])
            show_render()
    else:
        st.info("This synthetic source has illustrated tracklets but no original video. Use a real processed source to inspect frames.")
    st.caption("Original SHA-256: " + (source.get("sha256") or "Synthetic"))
    display = st.radio("Results layout", ["Grid", "List", "Timeline"], horizontal=True)
    filtered.sort(key=lambda t: (t["start_time"], t["id"]))
    if display == "Timeline" and filtered:
        rows = [{"Track": t["id"][-8:], "Start": t["start_time"], "End": t["end_time"], "Class": t.get("object_class", "person")} for t in filtered]
        figure = px.scatter(pd.DataFrame(rows), x="Start", y="Track", color="Class", hover_data=["End"], color_discrete_sequence=palette())
        figure.update_layout(height=max(260, min(800, len(rows) * 25)))
        plot(figure, use_container_width=True)
    elif display == "List":
        st.dataframe([{"Track": t["id"], "Class": t.get("object_class", "person"), "Colour": t.get("attributes", {}).get("color", "unknown"),
                       "Start (s)": t["start_time"], "End (s)": t["end_time"], "Confidence": t.get("avg_conf"),
                       "Visible duration (s)": t.get("duration_s")} for t in filtered], hide_index=True, use_container_width=True)
    else:
        for offset in range(0, min(len(filtered), 30), 3):
            for col, track in zip(st.columns(3), filtered[offset:offset+3]):
                with col, st.container(border=True):
                    show_track(service, track, source, actor, "vr_" + track["id"])
                    st.caption(f"{track.get('object_class', 'person').title()} · {track.get('attributes', {}).get('color', 'unknown')} · detection confidence {track.get('avg_conf', 0):.2f}")
                    if st.button("Bookmark", key="vr_book_" + track["id"]):
                        repo.put("bookmark", {"case_id": cid, "track_id": track["id"], "actor": actor}, actor)
                        st.success("Bookmarked")
        if len(filtered) > 30:
            st.caption("Showing the first 30 cards. Use filters or list view to narrow the results.")


def review(service, case, actor):
    repo, cid = service.repo, case["id"]
    st.write("Compare the two sightings, check the scores below, then record your decision. A suggested link needs your review.")
    with st.expander("Generate a new proposal run"):
        c1, c2, c3 = st.columns(3)
        similarity = c1.slider("Minimum raw cosine", -1., 1., .55, .01)
        margin = c2.slider("Minimum competitor margin", 0., 1., .08, .01)
        quality = c3.slider("Minimum crop quality", 0., 1., .30, .01)
        top_k = c1.slider("Candidates per source track", 1, 10, 5)
        experiments = {e["id"]: e for e in repo.list(cid, "experiment") if bool(e["artifact"]["synthetic"]) == bool(case["demo"])}
        classifier = st.selectbox("Pairwise classifier", [None] + list(experiments), format_func=lambda k: "Heuristic (uncalibrated)" if k is None else experiments[k]["report"]["selected_model"] + " / " + k[-8:])
        st.caption("New runs preserve previous proposals and decisions. Heuristic thresholds require target-data validation.")
        if st.button("Run ThreadSight ranking", disabled=case["status"] not in {"Under Review", "Reviewed"}):
            service.rank(cid, actor, top_k=top_k, policy=RejectionPolicy(similarity, quality, margin), classifier=experiments[classifier]["artifact"] if classifier else None)
            st.rerun()
    proposals, latest = repo.active_proposals(cid), repo.latest_reviews(cid)
    tracks = {t["id"]: t for t in repo.list(cid, "track")}
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    requested_proposal = st.session_state.pop("investigator_proposal", None)
    if requested_proposal:
        st.session_state["ts_review_state"] = "All"
        st.session_state["ts_model_state"] = "All"
    f1, f2 = st.columns(2)
    hf = f1.selectbox("Review state", ["All", "pending", "accept", "reject", "defer"], key="ts_review_state")
    mf = f2.selectbox("Model assessment", ["All", "review_priority", "uncertain", "no_reliable_candidate"], key="ts_model_state")
    selected = [p for p in proposals if (hf == "All" or latest.get(p["id"], {}).get("decision", "pending") == hf) and (mf == "All" or p["verdict"]["state"] == mf)]
    selected.sort(key=lambda p: (latest.get(p["id"], {}).get("decision") in {"accept", "reject"},
                                 -(p["probability"] if p.get("probability") is not None else p["ranking"]["score"])))
    if not selected:
        st.info("No proposals in this view. Process two sources or adjust the filters.")
        return
    index_key = f"idx_{cid}_{hf}_{mf}_{len(selected)}"
    if requested_proposal:
        match = next((i for i, p in enumerate(selected) if p["id"] == requested_proposal), None)
        if match is not None:
            st.session_state[index_key] = match + 1
    index = st.number_input("Proposal number", 1, len(selected), 1, key=index_key) - 1
    proposal = selected[index]
    st.subheader(f"Candidate {index + 1} of {len(selected)}")
    st.caption(proposal["id"] + " · " + latest.get(proposal["id"], {}).get("decision", "pending"))
    for col, tid in zip(st.columns(2), [proposal["left_id"], proposal["right_id"]]):
        with col, st.container(border=True):
            show_track(service, tracks[tid], sources[tracks[tid]["source_id"]], actor, proposal["id"] + tid, True)
    f = proposal["features"]
    review_record = latest.get(proposal["id"], {})
    decision = review_record.get("decision", "pending")
    if decision == "pending":
        st.caption("Awaiting a reviewer decision")
    else:
        decision_label = {"accept": "Link accepted", "reject": "Link rejected", "defer": "Decision deferred"}[decision]
        st.markdown(render_custody_stamp(review_record.get("actor", ""), decision_label,
                                          review_record.get("created_at", ""), stamp_color="red" if decision == "reject" else "teal"),
                    unsafe_allow_html=True)
    if proposal.get("probability") is not None:
        st.metric("Pair classifier probability", f"{proposal['probability']:.1%}")
    score_help = [
        "Cosine similarity compares appearance vectors on a scale from −1 to 1. Higher means more similar in the model's feature space. 0.798 does not mean a 79.8% chance of being the same person.",
        "The lower image-quality score of the two sightings, from 0 to 1. It summarizes crop size, sharpness, lighting and shape. Higher usually means more usable imagery; it is not matching accuracy.",
        "How far this pair's ranking score is ahead of its strongest alternative, considering both sides. Positive means it leads, near zero means close competition, and negative means another candidate scored higher. It is not a probability.",
        "Checks configured camera connections and recording times. Plausible means a route fits the timing; implausible means known routes conflict; unknown means the map or time information is insufficient. It does not establish identity.",
    ]
    for col, label, value, help_text in zip(st.columns(4), ["Appearance similarity", "Lower image quality", "Lead over alternatives", "Route compatibility"],
                                 [f"{f['values']['similarity']:.3f}", f"{f['values']['quality_min']:.2f}", f"{f['values']['margin']:.3f}", f["route"]["status"]], score_help):
        col.metric(label, value, help=help_text)
    with st.expander("Score legend · how to read this comparison", expanded=st.session_state.get("beginner_guidance", True)):
        for title, explanation in zip(["Appearance similarity (cosine)", "Lower image quality", "Lead over alternatives (competitor margin)", "Route compatibility"], score_help):
            st.markdown(f"**{title}** — {explanation}")
        if f["values"]["margin"] < 0:
            st.warning("For this comparison, another candidate scored higher. Inspect the alternatives before deciding.")
        if f["route"]["status"] == "unknown":
            st.info("Route could not be assessed: " + f["route"]["reason"])
        st.caption("Accept = record that this link is supported by your review. Reject = record that you do not support this link. Defer = leave it unresolved until more evidence is available. Your decision does not establish a person's real-world identity.")
    if proposal["probability"] is not None:
        st.caption(f"Calibrated classifier estimate: {proposal['probability']:.3f} · {'synthetic experiment only' if case['demo'] else 'check evaluation and domain before interpreting'}")
    for reason in proposal["verdict"]["reasons"]:
        st.write("• " + reason)
    with st.expander("Features, quality and prior decisions"):
        st.json({"features": f, "ranking": proposal["ranking"], "model_version": proposal["model_version"], "run_id": proposal["run_id"]})
        for tid in (proposal["left_id"], proposal["right_id"]):
            st.json({"track_id": tid, "quality": tracks[tid].get("quality_report"), "embedding": tracks[tid].get("embedding_meta")})
        st.dataframe([r for r in repo.list(cid, "review") if r["proposal_id"] == proposal["id"]], use_container_width=True)
    with st.form("review_" + proposal["id"]):
        decision = st.radio("Human decision", ["defer", "accept", "reject"], horizontal=True)
        reason = st.text_area("Decision reason (required)")
        if st.form_submit_button("Record review", type="primary", disabled=case["status"] != "Under Review"):
            repo.review(cid, proposal["id"], actor, decision, reason)
            st.rerun()
    if st.button("Bookmark candidate"):
        repo.put("bookmark", {"case_id": cid, "proposal_id": proposal["id"], "actor": actor}, actor)
        st.success("Bookmark saved.")


def search_digest(service, case, actor):
    repo, cid = service.repo, case["id"]
    tracks = service.active_tracks(cid)
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    if tracks:
        with st.expander("Find similar tracklets"):
            from .review_video import similar_tracks
            reference = st.selectbox("Reference tracklet", tracks,
                                     format_func=lambda t: f"{sources[t['source_id']]['name']} · {t['id'][-8:]} · {t['start_time']:.1f}s")
            if st.button("Find similar appearances"):
                st.session_state["similar_" + cid] = (reference["id"], [(t["id"], score) for t, score in similar_tracks(reference, tracks)])
            prior = st.session_state.get("similar_" + cid)
            if prior and prior[0] == reference["id"]:
                by_id = {t["id"]: t for t in tracks}
                if not prior[1]:
                    st.info("No compatible indexed descriptors were found for this tracklet.")
                for tid, score in prior[1]:
                    if tid in by_id:
                        candidate = by_id[tid]
                        with st.container(border=True):
                            c_cand1, c_cand2 = st.columns([3, 1])
                            with c_cand1:
                                show_track(service, candidate, sources[candidate["source_id"]], actor, "similar_" + tid, True)
                            with c_cand2:
                                st.metric("Appearance similarity", f"{score:.3f}")

    c1, c2, c3, c4 = st.columns(4)
    sid = c1.selectbox("Source", [None] + list(sources), format_func=lambda x: sources[x]["name"] if x else "All sources")
    color = c2.selectbox("Approximate colour", ["All"] + sorted({t.get("attributes", {}).get("color", "unknown") for t in tracks}))
    quality = c3.slider("Minimum quality", 0., 1., 0., .05)
    motion = c4.selectbox("Motion", ["All", "Stationary", "Moving"])
    category = st.selectbox("Object category", ["All", "person", "vehicle", "bag"])
    c5, c6, c7 = st.columns(3)
    upper = c5.selectbox("Approximate upper clothing colour", ["All"] + sorted({t.get("attributes", {}).get("upper_color", "unknown") for t in tracks if t.get("object_class", "person") == "person"}))
    lower = c6.selectbox("Approximate lower clothing colour", ["All"] + sorted({t.get("attributes", {}).get("lower_color", "unknown") for t in tracks if t.get("object_class", "person") == "person"}))
    subtype_names = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    subtype = c7.selectbox("Vehicle subtype", ["All"] + sorted({subtype_names.get(t.get("class_id"), "unknown") for t in tracks if t.get("object_class") == "vehicle"}))
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Clip interval start (s)", min_value=0., value=0.)
    end = c2.number_input("Clip interval end (s)", min_value=0., value=600.)
    duration = c3.number_input("Minimum visible duration (s)", min_value=0., value=0.)
    if end < start:
        raise ValueError("Search end time must follow start time.")
    result = [t for t in tracks if (sid is None or t["source_id"] == sid) and (category == "All" or t.get("object_class", "person") == category) and (color == "All" or t.get("attributes", {}).get("color") == color)
              and (upper == "All" or t.get("attributes", {}).get("upper_color") == upper) and (lower == "All" or t.get("attributes", {}).get("lower_color") == lower)
              and (subtype == "All" or subtype_names.get(t.get("class_id"), "unknown") == subtype)
              and t.get("embedding_meta", {}).get("mean_quality", 0) >= quality and t["end_time"] >= start and t["start_time"] <= end
              and t.get("duration_s", 0) >= duration and (motion == "All" or t.get("motion", {}).get("stationary", False) == (motion == "Stationary"))]
    st.caption(f"{len(result)} tracklets · clothing colours are rough crop-region estimates; bag/hat classifiers are not configured.")
    page = st.number_input("Digest page", 1, max(1, (len(result) + 11) // 12), 1)
    for offset in range((page - 1) * 12, min(page * 12, len(result)), 3):
        for col, track in zip(st.columns(3), result[offset:offset + 3]):
            with col, st.container(border=True):
                show_track(service, track, sources[track["source_id"]], actor, "digest_" + track["id"], True)
                st.caption("Source SHA-256: " + (sources[track["source_id"]].get("sha256") or "Synthetic"))
                if st.button("Bookmark tracklet", key="bt_" + track["id"]):
                    repo.put("bookmark", {"case_id": cid, "track_id": track["id"], "actor": actor}, actor)
                    st.success("Saved")
    st.subheader("Review reel")
    st.caption("A playable sequence of indexed preview crops with source and clip times. Separate sightings remain separate; open original context to verify them.")
    reel_key = "reel_" + cid
    reel_selection = tuple(t["id"] for t in result[:60])
    if st.button("Prepare review reel", disabled=not result):
        from .digest import review_reel
        st.session_state[reel_key] = (reel_selection, review_reel(service, result, sources, actor))
    if reel_key in st.session_state and st.session_state[reel_key][0] == reel_selection:
        st.video(st.session_state[reel_key][1])
        st.download_button("Download review reel", st.session_state[reel_key][1], f"fedsight-reel-{cid[-8:]}.mp4", "video/mp4")
    st.subheader("Case export")
    st.caption("PDF report, exact frame/time references, thumbnails, CSVs, review history and audit checkpoint. The downloaded ZIP is plaintext; store it securely.")
    if st.button("Verify and prepare case bundle", type="primary"):
        from .reports import export_bundle
        st.session_state["export_" + cid] = export_bundle(service, cid, actor)
    if "export_" + cid in st.session_state:
        st.download_button("Download prepared case snapshot", st.session_state["export_" + cid], f"fedsight-{cid[-8:]}.zip", "application/zip")
        st.caption("Prepare again to include decisions made after this snapshot.")


def _legacy_investigation_chat(service, case, actor):
    from .investigation_chat import MODELS, evidence_time, local_candidates, vision_rerank
    repo, cid = service.repo, case["id"]
    tracks = service.active_tracks(cid)
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    for source_id, alignment in service.blind_alignments(cid).items():
        sources[source_id] = {**sources[source_id], "camera_id": alignment["camera_id"],
                              "time_anchor": alignment["time_anchor"]}
    cameras = {c["id"]: c["label"] for c in repo.list(cid, "camera")}
    jobs = {j["id"]: j for j in repo.list(cid, "job")}
    proposals = repo.active_proposals(cid)
    reviews = repo.latest_reviews(cid)
    st.write("Describe what you are looking for. Results link back to indexed preview crops and their original recordings.")
    st.caption("A visual match is an investigative lead, not a confirmed identity. Timestamps come from video metadata and documented clock alignment, never from the language model.")
    mode = st.radio("Search engine", ["Local metadata · free", "OpenAI visual search · API key", "ChatGPT plan · sign in"], horizontal=True)
    api_key = None
    model = None
    max_images = 16
    scan_start = 1
    max_results = 12
    send_previews = False
    store = None
    if mode.startswith("ChatGPT"):
        from .siwc import CredentialStore, available_models, finish_sign_in, start_sign_in
        store = CredentialStore(service.keys.directory, service.keys.encryption)
        connected = store.load()
        if connected:
            st.success("Connected ChatGPT account: " + (connected.get("email") or connected["subject"][:12]))
            if st.button("Disconnect ChatGPT account"):
                try:
                    store.disconnect()
                except Exception:
                    st.warning("Local credentials were removed; remote revocation could not be confirmed.")
                st.session_state.pop("siwc_models", None)
                st.rerun()
            if st.button("Check models available to this account"):
                try:
                    st.session_state["siwc_models"] = [model_id for model_id in MODELS if model_id in available_models(store)]
                except Exception as exc:
                    st.error(f"Could not list models: {type(exc).__name__}")
        else:
            st.caption("Eligible Plus accounts can authorize this locally hosted workbench through OpenAI. No API key is needed for this mode.")
            attempt_key = "siwc_attempt"
            if attempt_key not in st.session_state and st.button("Start ChatGPT sign-in"):
                st.session_state[attempt_key] = start_sign_in(store)
                st.rerun()
            if attempt_key in st.session_state:
                attempt = st.session_state[attempt_key]
                st.link_button("Continue with ChatGPT", attempt.authorization_url, type="primary")
                @st.fragment(run_every=2)
                def poll_sign_in():
                    try:
                        result = finish_sign_in(attempt, store)
                    except Exception as exc:
                        st.session_state.pop(attempt_key, None)
                        st.error(f"Sign-in failed: {type(exc).__name__}: {str(exc)[:200]}")
                        return
                    if result:
                        st.session_state.pop(attempt_key, None)
                        st.session_state.pop("siwc_models", None)
                        st.rerun()
                poll_sign_in()
        permitted_models = st.session_state.get("siwc_models", list(MODELS))
        if connected and not permitted_models:
            st.warning("This account does not currently list the supported vision models. Check access in ChatGPT settings.")
        model = st.selectbox("Vision model", permitted_models or list(MODELS), key="siwc_model")
        max_images = st.slider("Maximum preview images to send", 4, 48, 16, 4, key="siwc_max_images")
        scan_start = st.number_input("Start at ranked candidate #", 1, max(1, len(tracks)), 1, key="siwc_scan_start")
        send_previews = st.checkbox("Send selected case preview crops to OpenAI for this search", key="siwc_send_previews")
        st.caption("Requires OpenAI authorization and eligible plan access. Requests use store=false and stream=true. Only selected resized preview crops and your description are sent.")
    elif mode.startswith("OpenAI"):
        model = st.selectbox("Vision model", MODELS)
        max_images = st.slider("Maximum preview images to send", 4, 48, 16, 4)
        scan_start = st.number_input("Start at ranked candidate #", 1, max(1, len(tracks)), 1, key="api_scan_start")
        api_key = os.environ.get("OPENAI_API_KEY") or st.text_input("OpenAI API key for this session", type="password")
        send_previews = st.checkbox("Send up to the selected number of case preview crops to OpenAI for this search")
        st.caption("Only selected preview crops and your description are sent. The original videos and case database stay local. Responses use store=false. API usage can incur charges; a Plus subscription does not itself provide an API key.")
        if not api_key:
            st.info("Set OPENAI_API_KEY outside source control or enter a key above. Install requirements-ai.txt to enable this mode.")
    else:
        max_results = st.slider("Results to show", 5, 40, 12)
    if not tracks:
        st.info("Process footage or load the synthetic demo to build a searchable case index.")
        return
    history_key = "investigation_chat_" + cid
    history = st.session_state.setdefault(history_key, [])
    prompt = st.chat_input("Example: person in a red jacket near 01:20, or someone carrying a bag")
    if prompt:
        if mode.startswith("OpenAI") and (not api_key or not send_previews):
            st.error("Provide an API key and authorize sending the selected previews before visual search.")
        elif mode.startswith("ChatGPT") and (not store.load() or not send_previews):
            st.error("Connect ChatGPT and authorize sending the selected previews before visual search.")
        else:
            try:
                candidates, filters = local_candidates(prompt, tracks, sources, limit=max(512, len(tracks)))
                scan_candidates = candidates[scan_start-1:]
                if mode.startswith("OpenAI") or mode.startswith("ChatGPT"):
                    with st.spinner("Inspecting selected preview crops with OpenAI vision…"):
                        if mode.startswith("ChatGPT"):
                            from openai import OpenAI
                            token = store.valid_access_token()
                            client = OpenAI(api_key=token, timeout=60., max_retries=0)
                            found = vision_rerank(prompt, scan_candidates, service, actor, "", model=model,
                                                  max_images=max_images, client=client, stream=True)
                        else:
                            found = vision_rerank(prompt, scan_candidates, service, actor, api_key, model=model, max_images=max_images)
                    found = found[:12]
                    scanned = min(max_images, sum(bool(i["track"].get("preview_asset")) for i in scan_candidates))
                    answer = (f"Inspected {scanned} preview crop(s), starting at ranked candidate {scan_start}, from {len(candidates)} metadata candidates. "
                              f"{len(found)} were marked matching or uncertain. Check original recordings and competing candidates. "
                              "Unscanned crops remain unassessed; change the start number to search the next batch.")
                else:
                    found = candidates[:max_results]
                    answer = (f"Showing {len(found)} of {len(candidates)} metadata candidate(s). Approximate colours and track motion are search aids, not visual confirmation."
                              if found else "No indexed tracklet met the available metadata filters. This does not establish absence from the recordings.")
                    if filters["needs_vision"]:
                        answer += " Local metadata cannot verify " + ", ".join(filters["visual_terms"]) + "; use visual search or inspect the crops."
                record = repo.put("chat_search", {"case_id": cid, "actor": actor, "query": prompt,
                                                  "engine": "chatgpt_plan" if mode.startswith("ChatGPT") else "openai_vision" if mode.startswith("OpenAI") else "local_metadata",
                                                  "model": model, "filters": filters,
                                                  "candidate_count": len(candidates), "scan_start": scan_start,
                                                  "result_track_ids": [i["track"]["id"] for i in found],
                                                  "job_ids": sorted({i["track"]["job_id"] for i in found}),
                                                  "preview_count_sent": min(max_images, sum(bool(i["track"].get("preview_asset")) for i in scan_candidates)) if not mode.startswith("Local") else 0,
                                                  "answer": answer}, actor)
                history.append({"question": prompt, "answer": answer, "record_id": record["id"],
                                "results": [{"track_id": i["track"]["id"], "verification": i["verification"],
                                             "reasons": i["reasons"], "visual_reason": i.get("visual_reason"),
                                             "model": i.get("model")} for i in found]})
                del history[:-20]
                st.rerun()
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Visual search failed ({type(exc).__name__}). Check API access and try again.")
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption("Search record: " + turn["record_id"])
    if not history or not history[-1]["results"]:
        return
    st.subheader("Latest source-linked results")
    all_tracks = {t["id"]: t for t in repo.list(cid, "track")}
    export_rows = []
    for item in history[-1]["results"]:
        track = all_tracks.get(item["track_id"])
        if track:
            source = sources[track["source_id"]]
            when = evidence_time(track, source, jobs.get(track["job_id"]))
            export_rows.append({"track_id": track["id"], "source_id": source["id"], "source_name": source["name"],
                                "source_sha256": source.get("sha256"), "preview_frame": when["preview_frame"],
                                "preview_clip_s": when["preview_clip_s"], "track_start_s": track["start_time"],
                                "track_end_s": track["end_time"], "aligned_utc": when["utc_midpoint"],
                                "assessment": item["verification"], "visual_reason": item["visual_reason"]})
    if export_rows:
        from .reports import csv_bytes
        st.download_button("Download result manifest", csv_bytes(export_rows),
                           f"fedsight-chat-{history[-1]['record_id'][-8:]}.csv", "text/csv")
    for position, item in enumerate(history[-1]["results"], 1):
        track = all_tracks.get(item["track_id"])
        if track is None:
            continue
        source = sources[track["source_id"]]
        when = evidence_time(track, source, jobs.get(track["job_id"]))
        with st.container(border=True):
            image_col, details_col = st.columns([1, 2])
            with image_col:
                if track.get("preview_asset"):
                    st.image(service.read_asset(track["preview_asset"], actor), caption="Indexed crop from recording", use_container_width=True)
            with details_col:
                st.markdown(f"**{position}. {source['name']} · {track.get('object_class', 'person').title()}**")
                st.write(f"Preview at clip **{when['preview_clip_s']:.2f} s** · visible track interval **{track['start_time']:.2f}–{track['end_time']:.2f} s**")
                st.caption(f"Frame: {when['preview_frame'] if when['preview_frame'] is not None else 'not stored'} · Camera: {cameras.get(source.get('camera_id'), 'unknown')} · Track: {track['id']}")
                if when["utc_midpoint"]:
                    st.caption(f"Aligned UTC midpoint: {when['utc_midpoint']} · uncertainty bounds: {when['utc_uncertainty_bounds']}")
                st.caption(f"Original SHA-256: {source.get('sha256') or 'synthetic demo'}")
                if item["visual_reason"]:
                    st.write(f"Model assessment ({item['verification']}): {item['visual_reason']}")
                else:
                    st.write("Metadata basis: " + (", ".join(item["reasons"]) or "broad indexed search"))
                related = [p for p in proposals if track["id"] in {p["left_id"], p["right_id"]}][:3]
                if related:
                    st.caption("Cross-camera candidates: " + "; ".join(f"{p['id'][-8:]} ({reviews.get(p['id'], {}).get('decision', 'pending')})" for p in related))
                if source.get("asset") and st.checkbox("Open original video context", key=f"chat_video_{history[-1]['record_id']}_{position}"):
                    st.video(service.read_asset(source["asset"], actor), start_time=max(0., when["preview_clip_s"]-3),
                             end_time=min(track["end_time"]+3, when["preview_clip_s"]+5))
                if st.button("Bookmark this sighting", key=f"chat_bookmark_{history[-1]['record_id']}_{position}"):
                    repo.put("bookmark", {"case_id": cid, "track_id": track["id"], "actor": actor,
                                          "chat_search_id": history[-1]["record_id"]}, actor)
                    st.success("Sighting bookmarked.")


def investigation_chat(service, case, actor):
    """Case-scoped investigator with durable, inspectable offline tool results."""
    from .investigator import InvestigatorTools, answer_offline
    from .investigator_engines import PROVIDERS, answer_with_engine, test_connection
    from .local_vision import QUEUE as VISION_QUEUE
    from .semantic_search import QUEUE as CLIP_QUEUE
    repo, cid = service.repo, case["id"]
    tools = InvestigatorTools(service, cid)
    st.write("Ask about indexed sightings, recorded events, similar appearances, or unobserved intervals.")
    st.caption("Visual match is a lead, not a confirmed identity. Answers use records from the active case.")
    if not service.active_tracks(cid):
        st.info("No sightings are indexed yet. Add and process a recording to search its appearances.")
        st.button("Add or process recordings →", key="investigator_add_video",
                  on_click=lambda: st.session_state.update(workspace="Evidence & processing"))
    sessions = repo.list(cid, "investigator_session")
    all_messages = repo.list(cid, "investigator_message")
    session_key = "investigator_session_" + cid
    new_chat = "__new_conversation__"
    requested_session = st.session_state.pop("open_investigator_session", None)
    if requested_session == new_chat or requested_session in {s["id"] for s in sessions}:
        st.session_state[session_key] = requested_session
    if st.session_state.get(session_key) not in {new_chat, *(s["id"] for s in sessions)}:
        st.session_state[session_key] = new_chat
    controls = st.columns([3, 1])
    labels = {new_chat: "New chat"}
    for session in sessions:
        first = next((m.get("question", "") for m in all_messages if m["session_id"] == session["id"]), "")
        title = " ".join(first.split())[:55] or "Untitled conversation"
        labels[session["id"]] = f"{title} · {session['created_at'][:19]} · {session['id'][-6:]}"
    chosen = controls[0].selectbox("Previous chats", [new_chat] + [s["id"] for s in reversed(sessions)],
                                   key=session_key, format_func=lambda value: labels[value],
                                   help="Choose a saved chat from this case, or select New chat to start fresh.")
    selected = None if chosen == new_chat else chosen
    if controls[1].button("New conversation"):
        st.session_state["open_investigator_session"] = new_chat
        st.session_state.pop("investigator_prompt_" + cid, None)
        st.rerun()
    messages = [m for m in all_messages if m["session_id"] == selected] if selected else []
    if selected is None:
        st.caption("New chat · Your previous conversations are available in the dropdown above.")
    with st.expander("Open-vocabulary visual search", expanded=False):
        semantic_enabled = st.checkbox("Search indexed preview images with MobileCLIP", value=False,
                                       help="Local text-to-image ranking; scores are not proof of a match.")
        clip_weights = st.text_input("Trusted MobileCLIP2-S0 checkpoint path", value=os.environ.get("FEDSIGHT_CLIP_WEIGHTS", ""),
                                     help="Provide locally downloaded pretrained MobileCLIP2-S0 weights.")
        st.caption("Vectors are stored in encrypted case records. Indexing can take time; after an app restart the in-memory search is rebuilt.")
        clip_key = (str(service.root.resolve()), cid)
        clip_task = CLIP_QUEUE.get(clip_key)
        clip_running = bool(clip_task and clip_task.snapshot()["status"] in {"queued", "running"})
        if st.button("Index current case previews", disabled=clip_running or not clip_weights):
            CLIP_QUEUE.start(service, cid, actor, clip_weights)
            st.rerun()
        @st.fragment(run_every=2 if clip_running else None)
        def show_clip_index():
            task = CLIP_QUEUE.get(clip_key)
            if task:
                state = task.snapshot()
                if state["status"] in {"queued", "running"}:
                    st.info(state["stage"])
                elif state["status"] == "failed":
                    st.error(state["error"])
                elif state["status"] == "completed":
                    st.success(f"Indexed {state['result']['indexed']} new previews; {state['result']['available']} active previews available.")
        show_clip_index()
        camera_options = {"All cameras": None, **{label: camera_id for camera_id, label in tools.cameras.items()}}
        clip_camera = st.selectbox("Filter by camera", list(camera_options))
        clip_filter_time = st.checkbox("Filter by recording-relative time")
        clip_start, clip_end = None, None
        if clip_filter_time:
            cstart, cend = st.columns(2)
            clip_start = cstart.number_input("From second", min_value=0., value=0.)
            clip_end = cend.number_input("To second", min_value=0., value=60.)
        clip_width_col, clip_height_col = st.columns(2)
        clip_min_width = clip_width_col.number_input("Minimum track box width (pixels)", min_value=0, value=0)
        clip_min_size = clip_height_col.number_input("Minimum track box height (pixels)", min_value=0, value=0)
        clip_top_k = st.slider("Maximum candidates", 1, 30, 8)
    working_set = messages[-1].get("working_set", []) if messages else []
    with st.expander("Reasoning engine", expanded=False):
        engine = st.selectbox("Choose engine", ["Built-in (offline)"] + list(PROVIDERS), index=0)
        local_vision = st.checkbox("Use local vision for visual details (slower)", value=True,
                                   help="Reviews person previews in batches with a local vision model (e.g. gemma4:12b via Ollama). Disable for an immediate response.")
        vision_model = "gemma4:12b"
        max_previews = 36
        if local_vision:
            v_col1, v_col2 = st.columns(2)
            vision_model = v_col1.text_input("Local vision model", value="gemma4:12b", key="inv_vision_model")
            max_previews = v_col2.number_input("Max previews to review", min_value=6, max_value=120, value=36, step=6, key="inv_vision_limit")
        engine_model = ""
        engine_key = None
        cloud_allowed = False
        if engine in PROVIDERS:
            config = PROVIDERS[engine]
            engine_model = st.text_input("Model", config["model"], key="investigator_model_" + engine)
            if config["env"]:
                try:
                    secret_key = st.secrets.get(config["env"])
                except Exception:
                    secret_key = None
                engine_key = os.environ.get(config["env"]) or secret_key or st.text_input("API key for this session", type="password", key="investigator_key_" + engine)
                cloud_allowed = st.checkbox("Send my question and camera labels to this cloud provider for tool selection", key="investigator_cloud_" + engine)
                st.caption("This provider only chooses a search tool. It does not see video or images; visual searches use local vision. Provider plans and free tiers may change.")
            else:
                st.caption("Requires Ollama running locally with the selected model. If unavailable, the built-in offline engine answers.")
            st.caption("Testing sends a short fixed test request to the selected provider and may use API quota. No case data is sent.")
            if st.button("Test selected model"):
                try:
                    test_connection(engine, engine_key, model=engine_model)
                    st.success("The selected model successfully completed a test tool call.")
                except Exception as exc:
                    st.warning(f"The selected model could not complete a test request ({type(exc).__name__}). Check the model name, API key, provider access and quota. Offline search remains available.")
        st.caption("All factual answers and citations are constructed from case records. The selected provider supplies only a tool choice.")
    if not tools.tracks and not tools.events:
        st.info("Load the synthetic walkthrough or process recordings to add searchable evidence.")
    with st.expander("Try a question", expanded=not messages):
        chips = st.columns(3)
        suggestions = ["Summarize this case", "Show restricted zone events", "Show the timeline"]
        for col, question in zip(chips, suggestions):
            if col.button(question, key="suggest_" + question, use_container_width=True):
                st.session_state["investigator_prompt_" + cid] = question
                st.rerun()
    if st.button("Investigate intrusions"):
        st.session_state["investigator_prompt_" + cid] = "Show restricted zone intrusion events"
        st.rerun()
    pending_key = "investigator_vision_pending_" + str(selected)
    pending = st.session_state.get(pending_key)
    prompt = st.chat_input("Ask about the active case…", disabled=bool(pending)) or st.session_state.pop("investigator_prompt_" + cid, None)
    if prompt:
        if not selected:
            created = repo.put("investigator_session", {"case_id": cid, "actor": actor}, actor)
            st.session_state["open_investigator_session"] = created["id"]
            st.session_state["investigator_prompt_" + cid] = prompt
            st.rerun()
        with st.spinner("Searching case evidence…"):
            from .investigator import parse_intent
            intent, _ = parse_intent(prompt, tools, working_set)
            semantic_intent = intent in {"clarify", "clarify_unsupported_appearance", "clarify_person_description", "clarify_colour", "search_tracklets"}
            if semantic_enabled and semantic_intent:
                try:
                    import time
                    semantic_started = time.perf_counter()
                    from .semantic_search import get_model, search_case
                    if clip_end is not None and clip_start > clip_end:
                        raise ValueError("The start time must be before the end time.")
                    model = get_model(clip_weights)
                    ranked = search_case(service, cid, model, prompt, limit=clip_top_k,
                                         camera_id=camera_options[clip_camera], start_s=clip_start, end_s=clip_end,
                                         min_width=clip_min_width if clip_min_width else None,
                                         min_height=clip_min_size if clip_min_size else None)
                    ids = [item["track"]["id"] for item in ranked["results"]]
                    if not ids:
                        raise ValueError("No indexed previews match these filters. Build the case index first or widen the filters.")
                    if local_vision:
                        queue_key = (str(service.root.resolve()), cid, selected)
                        VISION_QUEUE.start(queue_key, prompt, tools, actor, model=vision_model, limit=len(ids), candidate_ids=ids)
                        st.session_state["investigator_vision_pending_" + str(selected)] = {"question": prompt, "key": queue_key, "model": vision_model}
                        st.rerun()
                    evidence = [{**tools.track_evidence(item["track"]), "clip_similarity": item["score"]} for item in ranked["results"]]
                    result = {"answer": f"Showing {len(evidence)} ranked candidates from {ranked['indexed']} indexed previews. Similarity is not a match probability; inspect the original video.",
                              "evidence": evidence, "working_set": ids, "engine": "MobileCLIP · local", "latency_ms": round((time.perf_counter() - semantic_started) * 1000),
                              "tool_calls": [{"name": "clip_search", "arguments": {"index_count": ranked["indexed"], "model_id": model.model_id, "vector_math_ms": ranked["search_ms"]}}]}
                except Exception as exc:
                    result = answer_offline(prompt, tools, working_set)
                    result["fallback"] = f"Open-vocabulary search unavailable: {str(exc)[:250]}"
            elif local_vision and intent == "clarify_unsupported_appearance":
                queue_key = (str(service.root.resolve()), cid, selected)
                VISION_QUEUE.start(queue_key, prompt, tools, actor, model=vision_model, limit=int(max_previews))
                st.session_state["investigator_vision_pending_" + str(selected)] = {"question": prompt, "key": queue_key, "model": vision_model}
                st.rerun()
            elif engine == "Built-in (offline)" or (engine in PROVIDERS and PROVIDERS[engine]["env"] and not cloud_allowed):
                result = answer_offline(prompt, tools, working_set)
                if engine != "Built-in (offline)":
                    result["fallback"] = "Cloud sharing was not enabled; used offline search."
            else:
                result = answer_with_engine(prompt, tools, working_set, provider=engine, model=engine_model, api_key=engine_key)
        repo.put("investigator_message", {"case_id": cid, "session_id": selected, "actor": actor,
                                          "question": prompt, **result}, actor)
        # Keep the existing search audit/export record for older integrations.
        repo.put("chat_search", {"case_id": cid, "actor": actor, "query": prompt, "engine": "local_metadata",
                                  "model": None, "result_track_ids": result["working_set"], "answer": result["answer"]}, actor)
        st.rerun()
    if pending:
        @st.fragment(run_every=2)
        def show_vision_progress():
            task = VISION_QUEUE.get(tuple(pending["key"]))
            if task is None:
                st.session_state.pop(pending_key, None)
                st.warning("The search was interrupted by an app restart. Please try again.")
                return
            state = task.snapshot()
            if state["status"] in {"queued", "running"}:
                st.info(f"Searching previews in the background for: {pending['question']} · {state['elapsed_s']:.0f}s elapsed. You can use other workspaces while it runs.")
                return
            if state["status"] == "completed":
                result = state["result"]
            else:
                result = answer_offline(pending["question"], tools, working_set)
                result["fallback"] = "Local visual search could not finish: " + (state.get("error") or "unknown error")
            repo.put("investigator_message", {"case_id": cid, "session_id": selected, "actor": actor,
                                              "question": pending["question"], **result}, actor)
            repo.put("chat_search", {"case_id": cid, "actor": actor, "query": pending["question"], "engine": "local_vision",
                                      "model": pending.get("model", "gemma4:12b"), "result_track_ids": result["working_set"], "answer": result["answer"]}, actor)
            st.session_state.pop(pending_key, None)
            st.rerun(scope="app")
        show_vision_progress()
    if messages:
        st.caption("Newest responses appear first.")
    for message in reversed(messages):
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
            st.caption(f"Reasoning engine: {message['engine']} · {message.get('latency_ms', 0)} ms")
            if message.get("fallback"):
                st.caption(message["fallback"])
            with st.expander("Evidence used"):
                for call in message.get("tool_calls", []):
                    st.json(call)
            a, b, c = st.columns(3)
            if a.button("Pin to case notes", key="pin_" + message["id"]):
                repo.put("note", {"case_id": cid, "text": message["answer"], "actor": actor,
                                  "investigator_message_id": message["id"]}, actor)
                st.success("Pinned to case notes.")
            if b.button("Helpful 👍", key="good_" + message["id"]):
                repo.put("investigator_feedback", {"case_id": cid, "message_id": message["id"], "rating": "helpful", "actor": actor}, actor)
                st.success("Feedback saved.")
            if c.button("Needs work 👎", key="bad_" + message["id"]):
                repo.put("investigator_feedback", {"case_id": cid, "message_id": message["id"], "rating": "needs_work", "actor": actor}, actor)
                st.success("Feedback saved.")
            with st.expander("Copy or regenerate"):
                st.code(message["answer"], language=None)
                if st.button("Regenerate answer", key="regen_" + message["id"]):
                    st.session_state["investigator_prompt_" + cid] = message["question"]
                    st.rerun()
            if message.get("reviewed_no_match"):
                with st.expander("Review images the model did not match", expanded=not message.get("evidence")):
                    st.caption("The model marked these as not matching. Inspect them manually: the model can make mistakes.")
                    for item in message["reviewed_no_match"]:
                        left, right = st.columns([1, 3])
                        if item.get("preview_asset"):
                            left.image(service.read_asset(item["preview_asset"], actor), use_container_width=True)
                        right.write(f"{item['source']} · {item['start_s']:.2f}–{item['end_s']:.2f}s")
                        right.caption("Not matched · " + item.get("visual_reason", ""))
            for evidence in message.get("evidence", [])[:24]:
                with st.container(border=True):
                    left, right = st.columns([1, 3])
                    if evidence.get("preview_asset"):
                        left.markdown(f'''
                        <div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--accent);letter-spacing:0.1em;margin-bottom:3px;">
                            // LEAD {evidence['id'][-8:].upper()}
                        </div>
                        ''', unsafe_allow_html=True)
                        left.image(service.read_asset(evidence["preview_asset"], actor), use_container_width=True)
                    right.markdown(f"**{evidence['type'].title()} · {evidence['id'][-10:]}**")
                    if evidence.get("source"):
                        right.write(f"{evidence.get('camera', 'Unknown camera')} · {evidence['source']} · clip {evidence['start_s']:.2f}–{evidence.get('end_s', evidence['start_s']):.2f}s")
                        clock = evidence.get("start_utc") or "unaligned"
                        if evidence.get("start_utc"):
                            clock += f" ±{evidence.get('clock_uncertainty_s', 0):g}s"
                        right.caption("UTC: " + clock + " · Source SHA-256: " + (evidence.get("source_hash") or "Synthetic"))
                    if evidence.get("rule"):
                        right.caption(f"{evidence['rule']} · {evidence.get('severity', 'medium')} · {evidence.get('review', 'pending')}")
                    if evidence.get("type") == "hypothesis":
                        right.write("Unobserved interval hypothesis · " + evidence.get("assessment", "unknown"))
                    if evidence.get("similarity") is not None:
                        right.caption(f"Cosine {evidence['similarity']:.3f} · not a probability")
                    if evidence.get("clip_similarity") is not None:
                        right.caption(f"Text-image similarity {evidence['clip_similarity']:.3f} · not a probability")
                    if evidence.get("visual_reason"):
                        right.caption("Local vision observation: " + evidence["visual_reason"])
                    if evidence.get("visual_verdict"):
                        right.caption("Assessment: " + ("Uncertain — review image" if evidence["visual_verdict"] == "uncertain" else "Possible match — verify in video"))
                    source = tools.sources.get(evidence.get("source_id"))
                    if source and source.get("asset") and st.checkbox("Show original clip", key="clip_" + message["id"] + evidence["id"]):
                        st.video(service.read_asset(source["asset"], actor), start_time=max(0., evidence["start_s"] - 3),
                                 end_time=evidence.get("end_s", evidence["start_s"]) + 3)
                        st.caption("Original recording, with clip-relative context. Use Video review for annotations.")
                    track_id = evidence.get("track_id")
                    actions = st.columns(3)
                    if evidence.get("source_id"):
                        def jump_to_clip(e=evidence):
                            st.session_state["investigator_jump"] = {
                                "case_id": cid, "source_id": e["source_id"], "seconds": e["start_s"],
                                "track_id": e.get("track_id")
                            }
                            st.session_state.workspace = "Video review"
                        actions[0].button("Jump to clip", key="jump_" + message["id"] + evidence["id"], on_click=jump_to_clip)
                    if track_id and actions[1].button("Add to case", key="add_" + message["id"] + evidence["id"]):
                        try:
                            repo.get(cid, track_id, "track")
                            repo.put("bookmark", {"case_id": cid, "track_id": track_id, "actor": actor,
                                                  "investigator_message_id": message["id"]}, actor)
                            st.success("Added to the case board.")
                        except Exception as exc:
                            st.warning(f"Could not bookmark sighting: {exc}")
                    linked = next((p for p in repo.active_proposals(cid) if track_id and track_id in {p["left_id"], p["right_id"]}), None)
                    if linked:
                        def jump_to_proposal(proposal_id=linked["id"]):
                            st.session_state["investigator_proposal"] = proposal_id
                            st.session_state.workspace = "ThreadSight review"
                        actions[2].button("Open in ThreadSight", key="thread_" + message["id"] + evidence["id"], on_click=jump_to_proposal)


def case_board(service, case, actor):
    """Show the active case's bookmarks, notes, and verifiable export."""
    from .reports import export_bundle
    notice = st.session_state.pop("case_manager_notice", None)
    if notice:
        st.success(notice)
    with st.expander("Create a new case", expanded=case is None):
        create_case_form(service, actor)
    repo = service.repo
    cid = case["id"] if case else None
    storage = repo.storage_info(cid)
    if case:
        with st.expander("Edit case details"):
            with st.form("case_details_" + cid):
                title = st.text_input("Case name", value=case["title"])
                description = st.text_area("Case description", value=case.get("description", ""))
                consent = st.text_area("Retention notes", value=case.get("consent", ""))
                if st.form_submit_button("Save case details"):
                    repo.edit_case(cid, title, description, consent, actor)
                    st.session_state.case_manager_notice = "Case details saved."
                    st.rerun()
    with st.expander("Storage and deletion", expanded=bool(storage["pending_files"])):
        mib = 1024 ** 2
        st.write(f"Shared SQLite database · {storage['database_bytes'] / mib:.2f} MiB, including "
                 f"{storage['unused_bytes'] / mib:.2f} MiB of reusable pages.")
        if st.button("Compact database", help="Return unused SQLite pages to disk. This applies to the whole workbench."):
            repo.compact()
            st.session_state.case_manager_notice = "Database compacted."
            st.rerun()
        if storage["pending_files"]:
            st.warning(f"{storage['pending_files']} encrypted files are awaiting removal. Close any program holding them, then retry.")
            if st.button("Retry file cleanup"):
                remaining = service.clean_deleted_assets()
                if remaining:
                    st.warning(f"{remaining} files still need cleanup.")
                else:
                    st.rerun()
        if case:
            counts = storage["counts"]
            assets = repo.list(cid, "asset")
            size = sum(a.get("bytes", 0) for a in assets)
            st.subheader("Delete active case")
            st.write(f"{case['title']} · {cid[-8:]} · {counts.get('source', 0)} recordings · "
                     f"{counts.get('investigator_session', 0)} chats · {len(assets)} evidence files "
                     f"({size / mib:.2f} MiB before encryption)")
            st.caption("Deletion permanently removes this case's recordings from the vault, previews, analyses, chats, "
                       "notes and reviews. The shared audit history retains identifiers, actors, timestamps and hashes. "
                       "Original uploads on your computer and exported copies stay where you saved them.")
            confirm = st.checkbox("I want to permanently delete this case", key="confirm_delete_" + cid)
            if st.button("Permanently delete case", disabled=not confirm, key="delete_" + cid):
                service.delete_case(cid, actor)
                live = st.session_state.get("live_" + cid)
                if live:
                    live.close()
                for key in list(st.session_state):
                    if cid in key:
                        del st.session_state[key]
                st.session_state.case_manager_notice = "Case deleted."
                st.rerun()
    if not case:
        st.info("Create a case to start a new investigation.")
        return
    st.write(case.get("description") or "Add investigation notes from Overview.")
    st.caption(f"Status: {case['status']} · Owner: {case.get('owner', 'unknown')}")
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    tracks = {t["id"]: t for t in repo.list(cid, "track")}
    proposals = {p["id"]: p for p in repo.list(cid, "proposal")}
    bookmarks = repo.list(cid, "bookmark")
    st.subheader("Evidence board")
    if not bookmarks:
        st.info("Bookmark tracklets from Video review or Investigation chat, or candidates from ThreadSight review.")
    for bookmark in reversed(bookmarks):
        ids = [bookmark["track_id"]] if bookmark.get("track_id") else []
        proposal = proposals.get(bookmark.get("proposal_id"))
        if proposal:
            ids = [proposal["left_id"], proposal["right_id"]]
        with st.container(border=True):
            st.markdown(render_custody_stamp(bookmark.get('actor', 'OFFICER'), "BOOKMARKED",
                                             bookmark.get('created_at', ''), stamp_color="teal"), unsafe_allow_html=True)
            st.caption(f"Bookmarked by {bookmark.get('actor', 'unknown')} · {bookmark.get('created_at', '')[:19]}")
            for column, tid in zip(st.columns(max(1, len(ids))), ids):
                track = tracks.get(tid)
                if track and track["source_id"] in sources:
                    with column:
                        show_track(service, track, sources[track["source_id"]], actor, "case_" + bookmark["id"] + tid, True)
                        st.caption("SHA-256: " + (sources[track["source_id"]].get("sha256") or "Synthetic"))
    st.subheader("Case notes")
    for note in reversed(repo.list(cid, "note")):
        st.write(note["text"])
        st.caption(f"{note['actor']} · {note['created_at'][:19]}")
    st.subheader("Case report and evidence bundle")
    st.caption("The bundle includes the PDF report, evidence hashes, review history, thumbnails and audit checkpoint.")
    if st.button("Prepare verified case report"):
        st.session_state["board_export_" + cid] = export_bundle(service, cid, actor)
    if "board_export_" + cid in st.session_state:
        st.download_button("Download case bundle", st.session_state["board_export_" + cid],
                           f"fedsight-case-{cid[-8:]}.zip", "application/zip")


def timeline(service, case, actor):
    from .graph import evidence_graph, constrained_query
    import networkx as nx
    repo, cid = service.repo, case["id"]
    tracks = service.active_tracks(cid)
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    if not tracks:
        st.info("Process evidence to build a timeline.")
        return
    shared = all(s.get("time_anchor", {}).get("original_timestamp") for s in sources.values())
    fig = go.Figure()
    for i, track in enumerate(tracks):
        source = sources[track["source_id"]]
        anchor = TimeAnchor(**source.get("time_anchor", {}))
        if shared:
            a, b = anchor.interval(track["start_time"]), anchor.interval(track["end_time"])
            x = [pd.to_datetime(sum(a) / 2, unit="s", utc=True), pd.to_datetime(sum(b) / 2, unit="s", utc=True)]
        else:
            x = [track["start_time"], track["end_time"]]
        fig.add_trace(go.Scatter(x=x, y=[source["name"] + " / " + track["id"][-5:]] * 2, mode="lines+markers", line=dict(width=8, color=palette()[i % len(palette())]), name=track["id"], showlegend=False, hovertext=f"Clock uncertainty ±{anchor.uncertainty_s}s"))
    fig.update_layout(height=max(360, 30 * len(tracks)), xaxis_title="Corrected UTC; uncertainty retained" if shared else "Clip-relative seconds — sources do not share a clock", margin=dict(l=10, r=10, t=10, b=20))
    plot(fig, use_container_width=True)
    if not shared:
        st.info("Source-relative lanes do not establish cross-camera chronology.")
    graph, groups, conflicts = evidence_graph(service, cid)
    st.subheader("Human-confirmed appearance groups")
    st.caption("Only accepted links form group edges. A model score never merges groups automatically.")
    for conflict in conflicts:
        st.warning(conflict["reason"])
    positions = nx.spring_layout(graph, seed=42)
    fig = go.Figure()
    for a, b in graph.edges:
        fig.add_trace(go.Scatter(x=[positions[a][0], positions[b][0]], y=[positions[a][1], positions[b][1]], mode="lines", line=dict(color=tokens()["muted"]), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in graph], y=[positions[n][1] for n in graph], mode="markers+text", text=[n[-6:] for n in graph], textposition="top center", marker=dict(size=16, color=palette()[0]), hovertext=list(graph), showlegend=False))
    fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=15, b=10))
    plot(fig, use_container_width=True)
    st.dataframe(groups, hide_index=True, use_container_width=True)
    st.subheader("Constrained evidence query")
    query = st.text_input("Question", placeholder="show pending proposals")
    st.caption("Supported: show pending proposals · show sources · show review history. Read-only templates.")
    if st.button("Query case records"):
        st.json(constrained_query(service, cid, query))
    with st.expander("Optional Neo4j projection"):
        st.caption("Sync sends decrypted case records to your configured Neo4j server. SQLite remains the authoritative audit store.")
        if st.button("Sync case to configured Neo4j"):
            from .graph import sync_neo4j
            if not all(os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")):
                raise ValueError("Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD outside source control first.")
            st.json(sync_neo4j(service, cid, os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))


def cameras(service, case, actor):
    repo, cid = service.repo, case["id"]
    records = repo.list(cid, "camera")
    mapping = {c["id"]: c["label"] for c in records}
    focus = None
    if records:
        focus = st.selectbox("Camera details", list(mapping), format_func=mapping.get, key="camera_focus_" + cid)
        camera = next(c for c in records if c["id"] == focus)
        st.caption(f"Site · {camera.get('site') or 'Unspecified'}   |   Zone · {camera.get('zone') or 'Unspecified'}")
    with st.form("camera"):
        st.subheader("Register camera")
        c1, c2, c3 = st.columns(3)
        label, site, zone = c1.text_input("Label"), c2.text_input("Site / floor"), c3.text_input("Zone")
        x, y = c1.number_input("Relative map X", value=0.), c2.number_input("Relative map Y", value=0.)
        direction = c3.text_input("View direction / region notes")
        calibration = st.text_input("Calibration version", "unverified-v1")
        if st.form_submit_button("Register camera"):
            service.register_camera(cid, label, actor, site=site, zone=zone, x=x, y=y, view_direction=direction, calibration_version=calibration)
            st.rerun()
    if len(records) >= 2:
        with st.form("transition"):
            st.subheader("Add directed transition")
            c1, c2 = st.columns(2)
            a = c1.selectbox("From camera", list(mapping), format_func=mapping.get)
            b = c2.selectbox("To camera", list(mapping), index=1, format_func=mapping.get)
            low = c1.number_input("Minimum travel seconds", min_value=0., value=10.)
            high = c2.number_input("Maximum travel seconds", min_value=0., value=120.)
            confidence = c1.slider("Configuration confidence", 0., 1., .8)
            provenance = st.text_input("Measurement / configuration provenance")
            if st.form_submit_button("Save directed transition"):
                service.add_transition(cid, actor, from_camera=a, to_camera=b, min_s=low, max_s=high, confidence=confidence, provenance=provenance)
                st.rerun()
    edges = repo.list(cid, "transition")
    if records:
        by_id, fig = {c["id"]: c for c in records}, go.Figure()
        for edge in edges:
            a, b = by_id[edge["from_camera"]], by_id[edge["to_camera"]]
            fig.add_annotation(x=b["x"], y=b["y"], ax=a["x"], ay=a["y"], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=tokens()["muted"], text="")
        fig.add_trace(go.Scatter(x=[c["x"] for c in records], y=[c["y"] for c in records], mode="markers+text", text=[c["label"] for c in records], textposition="top center", marker=dict(size=20, color=[palette()[1] if c["id"] == focus else palette()[0] for c in records])))
        fig.update_layout(height=350, xaxis_title="Relative map X", yaxis_title="Relative map Y", showlegend=False)
        plot(fig, use_container_width=True)
    st.caption("Routes are configuration hypotheses. Missing paths produce unknown feasibility, not impossible movement.")
    if edges:
        st.dataframe([{"From": mapping[e["from_camera"]], "To": mapping[e["to_camera"]], "Min seconds": e["min_s"], "Max seconds": e["max_s"], "Provenance": e["provenance"]} for e in edges], hide_index=True)


def blindsight(service, case, actor):
    repo, cid = service.repo, case["id"]
    cameras = {c["id"]: c for c in repo.list(cid, "camera")}
    regions = repo.list(cid, "coverage_region")
    zones = {r["id"]: r for r in regions if r["kind"] == "blind"}
    portals = repo.list(cid, "blind_portal")
    st.write("Map camera visibility and blind zones, then examine possible unobserved intervals between recorded person sightings.")
    st.caption("Video-only hypotheses. A missing interval does not establish presence in an unseen area or confirm two sightings are the same person.")
    source_records = {s["id"]: s for s in repo.list(cid, "source")}
    alignments = service.blind_alignments(cid)
    if source_records and cameras and st.session_state.get("mode") != "live":
        st.subheader("Align existing recordings")
        st.caption("This creates a versioned BlindSight mapping; the original video and its registration metadata remain preserved.")
        align_source = st.selectbox("Recording to align", list(source_records), format_func=lambda x: source_records[x]["name"])
        existing_alignment = alignments.get(align_source, source_records[align_source])
        anchor = existing_alignment.get("time_anchor", {})
        camera_options = list(cameras)
        default_camera = existing_alignment.get("camera_id")
        with st.form("blind_alignment"):
            align_camera = st.selectbox("Camera for this recording", camera_options,
                                        index=camera_options.index(default_camera) if default_camera in camera_options else 0,
                                        format_func=lambda x: cameras[x]["label"])
            timestamp = st.text_input("Recording start with UTC offset (optional)", anchor.get("original_timestamp") or "",
                                      help="Example: 2026-01-01T10:00:00+05:30. Leave empty if clocks are unaligned.")
            c1, c2, c3 = st.columns(3)
            offset = c1.number_input("Clock correction (s)", value=float(anchor.get("offset_s", 0.)))
            uncertainty = c2.number_input("Uncertainty ± (s)", min_value=0., value=float(anchor.get("uncertainty_s", 0.)))
            drift = c3.number_input("Clock drift (ppm)", value=float(anchor.get("drift_ppm", 0.)))
            method = st.text_input("Clock alignment method", anchor.get("method", "unknown"))
            align_provenance = st.text_input("Alignment evidence / provenance")
            if st.form_submit_button("Save source alignment"):
                service.set_blind_alignment(cid, actor, source_id=align_source, camera_id=align_camera,
                                            time_anchor={"original_timestamp": timestamp.strip() or None,
                                                         "timezone_name": "offset in timestamp" if timestamp.strip() else "unknown",
                                                         "offset_s": offset, "uncertainty_s": uncertainty,
                                                         "drift_ppm": drift, "method": method.strip() or "unknown"},
                                            provenance=align_provenance)
                st.rerun()
        st.dataframe([{"Source": source_records[sid]["name"], "Camera": cameras[a["camera_id"]]["label"],
                       "Clock start": a["time_anchor"].get("original_timestamp") or "unknown",
                       "Uncertainty (s)": a["time_anchor"].get("uncertainty_s"), "Provenance": a["provenance"]}
                      for sid, a in alignments.items()], hide_index=True, use_container_width=True)
    st.subheader("Coverage map")
    region_kind = st.selectbox("Map region type", ["visible", "blind", "obstacle", "occlusion"])
    with st.form("blind_region"):
        label = st.text_input("Region label")
        camera_id = st.selectbox("Visible from camera", [None] + list(cameras),
                                 format_func=lambda x: cameras[x]["label"] if x else "No camera / not applicable") if region_kind == "visible" else None
        bounds = [col.number_input(name, 0., 1., value, key="blind_region_" + name)
                  for col, name, value in zip(st.columns(4), ["Left", "Top", "Right", "Bottom"], [.2, .2, .8, .8])]
        distance = st.number_input("Measured path distance through zone (m, optional)", 0., 10000., 0.) if region_kind == "blind" else 0.
        provenance = st.text_input("Coverage source / measurement provenance")
        if st.form_submit_button("Save map region"):
            service.add_coverage_region(cid, actor, kind=region_kind, label=label, box=bounds,
                                        camera_id=camera_id, distance_m=distance or None, provenance=provenance)
            st.rerun()
    if zones and cameras:
        with st.form("blind_portal"):
            st.subheader("Mark a camera-edge portal")
            st.caption("Portal coordinates refer to the camera image, not the floor map. Mark where a person leaves or enters a blind zone.")
            zone_id = st.selectbox("Connected blind zone", list(zones), format_func=lambda x: zones[x]["label"])
            portal_camera = st.selectbox("Camera", list(cameras), format_func=lambda x: cameras[x]["label"])
            role = st.selectbox("Observed direction", ["both", "exit", "entry"])
            portal_bounds = [col.number_input(name, 0., 1., value, key="blind_portal_" + name)
                             for col, name, value in zip(st.columns(4), ["Image left", "Image top", "Image right", "Image bottom"], [.0, .2, .2, .8])]
            portal_provenance = st.text_input("Portal mapping provenance")
            if st.form_submit_button("Save portal"):
                service.add_blind_portal(cid, actor, camera_id=portal_camera, blind_zone_id=zone_id,
                                         role=role, box=portal_bounds, provenance=portal_provenance)
                st.rerun()
    hypotheses = service.active_blind_hypotheses(cid)
    fig = go.Figure()
    region_colors = {"visible": translucent(tokens()["accent"], .25), "blind": translucent(tokens()["warn"], .4),
                     "obstacle": translucent(tokens()["muted"], .4), "occlusion": translucent(tokens()["tertiary"], .35)}
    counts = {z: sum(any(r["zone_id"] == z for r in h.get("zones", [])) for h in hypotheses) for z in zones}
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        label = region["label"] + (f" · {counts[region['id']]} candidates" if region["kind"] == "blind" else "")
        fill = (translucent(tokens()["warn"], min(.2 + .12 * counts[region["id"]], .8))
                if region["kind"] == "blind" else region_colors[region["kind"]])
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=tokens()["muted"]), fillcolor=fill)
        fig.add_annotation(x=(x1+x2)/2, y=(y1+y2)/2, text=label, showarrow=False, font=dict(size=11))
    fig.update_layout(height=400, xaxis=dict(range=[0, 1], title="Schematic floor X", scaleanchor="y"),
                      yaxis=dict(range=[1, 0], title="Schematic floor Y"), showlegend=False,
                      margin=dict(l=20, r=20, t=10, b=20))
    plot(fig, use_container_width=True)
    st.caption("Schematic coordinates are configured by the investigator. Candidate density is a count of hypotheses, not measured occupancy.")
    if regions:
        st.dataframe([{"Region": r["label"], "Type": r["kind"], "Camera": cameras.get(r.get("camera_id"), {}).get("label", "—"),
                       "Distance (m)": r.get("distance_m"), "Provenance": r["provenance"]} for r in regions],
                     hide_index=True, use_container_width=True)
    if portals:
        st.caption(f"{len(portals)} image-plane portals configured")
    if st.session_state.get("mode") == "live":
        return
    st.subheader("Investigate missing intervals")
    proposals = repo.active_proposals(cid)
    if not proposals:
        st.info("Process at least two recordings and generate ThreadSight proposals first.")
    if not zones:
        st.info("Add a blind region and camera-edge portals to test a specific unseen route.")
    c1, c2 = st.columns(2)
    minimum_speed = c1.number_input("Minimum walking speed (m/s)", .1, 4., .5, .1)
    maximum_speed = c2.number_input("Maximum walking speed (m/s)", .1, 4., 2., .1)
    if st.button("Generate video-only hypotheses", type="primary", disabled=not proposals):
        service.generate_blind_hypotheses(cid, actor, min_speed_mps=minimum_speed, max_speed_mps=maximum_speed)
        st.rerun()
    if not hypotheses:
        return
    source_map = {s["id"]: s for s in repo.list(cid, "source")}
    tracks = {t["id"]: t for t in repo.list(cid, "track")}
    st.dataframe([{"Hypothesis": h["id"][-8:], "Assessment": h["assessment"],
                   "Zone": ", ".join(z["label"] for z in h["zones"]) or "unmapped",
                   "Gap seconds": "unknown" if h["gap_duration_s"] is None else f"{h['gap_duration_s'][0]:.1f}–{h['gap_duration_s'][1]:.1f}",
                   "Conflicts": len(h["contradictory_evidence"])} for h in hypotheses], hide_index=True, use_container_width=True)
    chosen = st.selectbox("Inspect hypothesis", hypotheses, format_func=lambda h: f"{h['id'][-8:]} · {h['assessment']}")
    st.caption("Possible unobserved movement · requires investigator review")
    for col, tid in zip(st.columns(2), [chosen["left_id"], chosen["right_id"]]):
        if tid in tracks:
            with col, st.container(border=True):
                show_track(service, tracks[tid], source_map[tracks[tid]["source_id"]], actor, "blind_" + chosen["id"] + tid, True)
    if chosen["gap_utc"]:
        end = chosen["gap_utc"]["last_seen_bounds"]
        start = chosen["gap_utc"]["next_seen_bounds"]
        fig = go.Figure(go.Scatter(x=[pd.to_datetime(sum(end)/2, unit="s", utc=True),
                                       pd.to_datetime(sum(start)/2, unit="s", utc=True)],
                                   y=["Unobserved interval"]*2, mode="lines+markers", line=dict(width=6, dash="dash", color=tokens()["warn"])))
        fig.update_layout(height=180, xaxis_title="Corrected UTC midpoint; see uncertainty bounds below", showlegend=False)
        plot(fig, use_container_width=True)
    st.markdown("**Supporting evidence**")
    for item in chosen["supporting_evidence"]:
        st.write("• " + item)
    st.markdown("**Contradictory or missing evidence**")
    for item in chosen["contradictory_evidence"]:
        st.write("• " + item)
    clues = [c for c in repo.list(cid, "blind_clue") if c["hypothesis_id"] == chosen["id"]]
    if clues:
        st.subheader("Investigator-recorded secondary clues")
        st.dataframe([{"Source": source_map[c["source_id"]]["name"], "Clip time": c["clip_time_s"],
                       "Type": c["category"], "Relationship": c["stance"], "Observation": c["description"],
                       "Provenance": c["provenance"]} for c in clues], hide_index=True, use_container_width=True)
    with st.form("blind_clue_" + chosen["id"]):
        st.markdown("**Record a secondary clue from original footage**")
        clue_source = st.selectbox("Recording", list(source_map), format_func=lambda x: source_map[x]["name"])
        clue_time = st.number_input("Clip-relative second", min_value=0., value=0.)
        clue_type = st.selectbox("Observed clue", ["door", "shadow", "reflection", "object_change", "audio", "other"])
        clue_stance = st.selectbox("Relationship to hypothesis", ["context_only", "supporting", "contradictory"])
        clue_description = st.text_area("What is visible or audible?")
        clue_provenance = st.text_input("How was this observation checked?")
        if st.form_submit_button("Add clue"):
            service.add_blind_clue(cid, actor, hypothesis_id=chosen["id"], source_id=clue_source,
                                   clip_time_s=clue_time, category=clue_type, stance=clue_stance,
                                   description=clue_description, provenance=clue_provenance)
            st.rerun()
    with st.expander("Route alternatives, time bounds and provenance"):
        st.json({"route": chosen["route"], "zones": chosen["zones"], "gap_utc": chosen["gap_utc"],
                 "disappearance": chosen.get("disappearance"), "reappearance": chosen.get("reappearance"),
                 "sources": [tracks[t]["source_id"] for t in (chosen["left_id"], chosen["right_id"]) if t in tracks],
                 "proposal_id": chosen["proposal_id"], "region_ids": chosen["region_ids"],
                 "portal_ids": chosen["portal_ids"], "transition_ids": chosen["transition_ids"],
                 "competing_proposal_ids": chosen.get("competing_proposal_ids", [])})
    decisions = [r for r in repo.list(cid, "blind_review") if r["hypothesis_id"] == chosen["id"]]
    if decisions:
        st.dataframe(decisions, hide_index=True, use_container_width=True)
    with st.form("blind_review_" + chosen["id"]):
        decision = st.selectbox("Investigator assessment", ["defer", "supports_possible_route", "discount_hypothesis"])
        reason = st.text_area("Reason and checked evidence")
        if st.form_submit_button("Record BlindSight review"):
            service.review_blind_hypothesis(cid, actor, chosen["id"], decision, reason)
            st.rerun()


def spatial_intelligence(service, case, actor):
    """Facility Digital Twin: 2.5D homography projections, trajectory mapping, and transit feasibility."""
    import base64
    from .spatial import (
        project_case_tracks,
        aggregate_heatmap,
        analyze_proposal_transit,
        transit_feasibility,
        classify_speed,
        load_homography,
        project_point,
    )
    repo, cid = service.repo, case["id"]
    st.caption("Planar homography projection, 2.5D floor-plan trajectories and physical transit-feasibility verification.")
    st.caption("Projected paths are estimated ground-plane positions. Feasibility scores assess physical locomotion plausibility; they do not establish identity.")

    calibrations = service.floor_calibrations(cid)
    tracks = service.active_tracks(cid)
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    cameras = {c["id"]: c for c in repo.list(cid, "camera")}
    jobs = {j["id"]: j for j in repo.list(cid, "job")}
    proposals = repo.active_proposals(cid)

    # Find floor plan asset
    floor_assets = [a for a in repo.list(cid, "asset") if a.get("role") == "floor_plan"]
    floor_asset = floor_assets[-1] if floor_assets else None
    if not floor_asset:
        st.info("Add a floor plan in the 4-Point Homography Calibration tab. Then select a recording and map four corresponding points to display its trajectories on the plan.")
    elif not calibrations:
        st.info("Floor plan added. Open the 4-Point Homography Calibration tab to map a recording onto it.")

    # Compute projected trajectories
    projected_tracks = project_case_tracks(calibrations, tracks, sources, jobs) if calibrations else {}

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CALIBRATED CAMERAS", len(calibrations))
    c2.metric("PROJECTED TRACKS", len(projected_tracks))
    c3.metric("FLOOR PLAN ASSETS", len(floor_assets))
    c4.metric("CROSS-CAMERA LEADS", len(proposals))

    tab_map, tab_feasibility, tab_calib = st.tabs([
        "Facility Map & Trajectories",
        "Transit Feasibility & Alibi Engine",
        "4-Point Homography Calibration"
    ])

    with tab_map:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            show_heatmap = st.checkbox("Overlay occupancy heatmap", value=False)
            show_all_tracks = st.checkbox("Show all projected trajectories", value=True)
        with col_ctrl2:
            track_choices = [t["id"] for t in tracks if t["id"] in projected_tracks]
            selected_track_id = st.selectbox("Inspect single track", ["None"] + track_choices,
                                             format_func=lambda tid: f"{tid[-8:]} · {sources.get(next((t['source_id'] for t in tracks if t['id'] == tid), ''), {}).get('name', '')}" if tid != "None" else "All / None")

        floor_w = floor_asset.get("width", 800) if floor_asset else 800
        floor_h = floor_asset.get("height", 450) if floor_asset else 450

        fig = go.Figure()

        # Add floor plan image background
        if floor_asset:
            try:
                raw_bytes = service.read_asset(floor_asset, actor)
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{b64}",
                        xref="x", yref="y",
                        x=0, y=0,
                        sizex=floor_w, sizey=floor_h,
                        sizing="stretch",
                        opacity=0.9,
                        layer="below"
                    )
                )
            except Exception:
                pass

        # Plot camera positions / calibration anchors
        for cal in calibrations:
            pts = cal.get("floor_points", [])
            if pts:
                xs = [p[0] for p in pts] + [pts[0][0]]
                ys = [p[1] for p in pts] + [pts[0][1]]
                src_name = sources.get(cal["source_id"], {}).get("name", "Camera")
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    name=f"FOV: {src_name}",
                    line=dict(color=tokens()["accent"], width=1, dash="dot"),
                    marker=dict(size=5, color=tokens()["accent"]),
                    hoverinfo="text",
                    text=[f"{src_name} FOV Anchor"] * len(xs)
                ))

        # Plot heatmap if enabled
        if show_heatmap and projected_tracks:
            grid = aggregate_heatmap(projected_tracks, floor_w, floor_h, cell_size=25)
            fig.add_trace(go.Heatmap(
                z=grid,
                x=np.linspace(0, floor_w, grid.shape[1]),
                y=np.linspace(0, floor_h, grid.shape[0]),
                colorscale=[tokens()["card"], tokens()["accent"]],
                opacity=0.45,
                showscale=False,
                hoverinfo="skip"
            ))

        # Plot trajectories
        tracks_to_plot = {}
        if selected_track_id != "None" and selected_track_id in projected_tracks:
            tracks_to_plot[selected_track_id] = projected_tracks[selected_track_id]
        elif show_all_tracks:
            tracks_to_plot = projected_tracks

        color_idx = 0
        path_colors = palette()
        for tid, points in tracks_to_plot.items():
            if not points:
                continue
            xs = [p["floor_x"] for p in points]
            ys = [p["floor_y"] for p in points]
            clr = path_colors[color_idx % len(path_colors)]
            color_idx += 1
            # Walking path line
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=f"Track {tid[-6:]}",
                line=dict(color=clr, width=3),
                hoverinfo="text",
                text=[f"Track {tid[-8:]} · {p['clip_s']:.1f}s · ({p['floor_x']:.0f}, {p['floor_y']:.0f})" for p in points]
            ))
            # Start and End points
            fig.add_trace(go.Scatter(
                x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
                mode="markers",
                name=f"Ends {tid[-6:]}",
                marker=dict(size=[8, 12], color=clr, symbol=["circle", "diamond"]),
                showlegend=False,
                hoverinfo="text",
                text=[f"Start: {tid[-8:]} @ {points[0]['clip_s']:.1f}s", f"End: {tid[-8:]} @ {points[-1]['clip_s']:.1f}s"]
            ))

        fig.update_xaxes(range=[0, floor_w], showgrid=True, gridcolor=tokens()["border"], zeroline=False, title="Floor X (px)")
        fig.update_yaxes(range=[floor_h, 0], showgrid=True, gridcolor=tokens()["border"], zeroline=False, title="Floor Y (px)")
        fig.update_layout(
            height=550,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=tokens()["text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        plot(fig, use_container_width=True)

    with tab_feasibility:
        st.subheader("Cross-Camera Transit & Alibi Analysis")
        st.write("Calculates required transit speeds between cross-camera track pairs. Human walking speeds exceed 2.0 m/s only when running; transit speeds > 8.0 m/s are physically impossible without vehicles.")

        track_dict = {t["id"]: t for t in tracks}
        ppm = calibrations[0].get("pixels_per_metre", 20.0) if calibrations else 20.0

        feasibility_rows = []
        for p in proposals:
            res = analyze_proposal_transit(p, track_dict, sources, calibrations, jobs, pixels_per_metre=ppm)
            if res:
                feasibility_rows.append(res)

        if feasibility_rows:
            feasibility_rows.sort(key=lambda r: r["speed_mps"])
            summary_cols = st.columns(3)
            likely_count = sum(r["feasibility"] in {"very_likely", "likely", "plausible"} for r in feasibility_rows)
            sprint_count = sum(r["feasibility"] == "possible" for r in feasibility_rows)
            implausible_count = sum(r["feasibility"] in {"unlikely", "implausible"} for r in feasibility_rows)

            summary_cols[0].metric("FEASIBLE LEADS (WALK)", likely_count)
            summary_cols[1].metric("REQUIRES SPRINT", sprint_count)
            summary_cols[2].metric("PHYSICALLY IMPLAUSIBLE", implausible_count, delta=-implausible_count if implausible_count else None)

            table_data = []
            for r in feasibility_rows:
                badge = "🟢 " if r["feasibility"] in {"very_likely", "likely"} else "🟡 " if r["feasibility"] == "plausible" else "🟠 " if r["feasibility"] == "possible" else "🔴 "
                table_data.append({
                    "Proposal": r["proposal_id"][-8:],
                    "Track A": r["left_id"][-8:],
                    "Track B": r["right_id"][-8:],
                    "Distance (m)": f"{r['distance_m']:.1f} m",
                    "Time Gap (s)": f"{r['delta_s']:.1f} s",
                    "Required Speed": f"{r['speed_mps']:.2f} m/s",
                    "Assessment": badge + r["label"]
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            chosen_prop = st.selectbox("Inspect transit vector", [r["proposal_id"] for r in feasibility_rows],
                                       format_func=lambda pid: f"Proposal {pid[-8:]}")
            prop_res = next(r for r in feasibility_rows if r["proposal_id"] == chosen_prop)

            v_col1, v_col2 = st.columns([1, 1])
            with v_col1:
                st.markdown(f"**Transit assessment:** {prop_res['label']}")
                st.write(f"Physical transit distance: **{prop_res['distance_m']:.2f} metres**")
                st.write(f"Elapsed interval: **{prop_res['delta_s']:.2f} seconds**")
                st.write(f"Implied velocity: **{prop_res['speed_mps']:.2f} m/s** ({prop_res['speed_mps']*3.6:.1f} km/h)")
                if prop_res["speed_mps"] > 8.0:
                    st.warning("⚠️ **Forensic Notice:** The subject would have to move faster than an Olympic sprinter to produce these sightings. This proposal is almost certainly a false-positive Re-ID coincidence.")
                else:
                    st.success("✅ **Forensic Notice:** Locomotion velocity is consistent with human transit between the two camera positions.")

            with v_col2:
                vec_fig = go.Figure()
                if floor_asset:
                    try:
                        raw_bytes = service.read_asset(floor_asset, actor)
                        b64 = base64.b64encode(raw_bytes).decode("ascii")
                        vec_fig.add_layout_image(
                            dict(
                                source=f"data:image/png;base64,{b64}",
                                xref="x", yref="y",
                                x=0, y=0,
                                sizex=floor_w, sizey=floor_h,
                                sizing="stretch",
                                opacity=0.8,
                                layer="below"
                            )
                        )
                    except Exception:
                        pass
                ea = prop_res["exit_floor"]
                eb = prop_res["entry_floor"]
                state_colours = {"very_likely":"positive", "likely":"positive", "plausible":"warn",
                                 "possible":"warn", "unlikely":"critical", "implausible":"critical"}
                route_colour = tokens()[state_colours.get(prop_res["feasibility"], "muted")]
                vec_fig.add_trace(go.Scatter(
                    x=[ea[0], eb[0]], y=[ea[1], eb[1]],
                    mode="lines+markers+text",
                    line=dict(color=route_colour, width=4, dash="solid" if prop_res["speed_mps"] <= 2.0 else "dash"),
                    marker=dict(size=[12, 14], color=[tokens()["accent"], route_colour], symbol=["circle", "arrow-bar-up"]),
                    text=["Exit A", "Entry B"],
                    textposition=["top center", "bottom center"],
                    hoverinfo="text",
                    hovertext=[f"Exit A: ({ea[0]:.0f}, {ea[1]:.0f})", f"Entry B: ({eb[0]:.0f}, {eb[1]:.0f})"]
                ))
                vec_fig.update_xaxes(range=[0, floor_w], showgrid=False, zeroline=False)
                vec_fig.update_yaxes(range=[floor_h, 0], showgrid=False, zeroline=False)
                vec_fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                plot(vec_fig, use_container_width=True)
        else:
            st.info("Calibrate at least two camera recordings to evaluate cross-camera physical transit feasibility.")

    with tab_calib:
        st.subheader("4-Point Homography Calibration")
        st.write("Calibrate the perspective transform between a video camera viewpoint and the facility floor plan.")

        with st.expander("Floor plan image management", expanded=not floor_assets):
            uploaded_fp = st.file_uploader("Upload new floor plan schematic (PNG or JPG)", type=["png", "jpg", "jpeg"], key="floor_upload_" + cid)
            if uploaded_fp and (floor_asset is None or digest(uploaded_fp.getvalue()) != floor_asset["sha256"]):
                img_data = uploaded_fp.getvalue()
                with Image.open(io.BytesIO(img_data)) as pil_img:
                    w, h = pil_img.size
                service.save_asset(img_data, cid, actor, role="floor_plan", parent_id=None,
                                   filename=uploaded_fp.name, width=w, height=h)
                st.rerun()
            if floor_asset:
                st.caption(f"Current floor plan · {floor_asset.get('filename', 'Case plan')} · {floor_asset['width']} × {floor_asset['height']} px")

        available_sources = list(sources.values())
        if available_sources:
            selected_source = st.selectbox("Select recording to calibrate", available_sources, format_func=lambda s: s["name"])
            sid = selected_source["id"]
            cam_id = selected_source.get("camera_id")

            existing_cal = next((c for c in calibrations if c["source_id"] == sid), None)
            if existing_cal:
                st.info(f"This recording is already calibrated ({existing_cal['provenance']}). Adding a new calibration will replace it.")

            c_pts1, c_pts2 = st.columns(2)
            with c_pts1:
                st.markdown("**Video frame 4 corners (pixels)**")
                v_p1_x = st.number_input("Point 1 X (Video)", value=50.0, step=10.0, key=f"v_p1_x_{sid}")
                v_p1_y = st.number_input("Point 1 Y (Video)", value=50.0, step=10.0, key=f"v_p1_y_{sid}")
                v_p2_x = st.number_input("Point 2 X (Video)", value=590.0, step=10.0, key=f"v_p2_x_{sid}")
                v_p2_y = st.number_input("Point 2 Y (Video)", value=50.0, step=10.0, key=f"v_p2_y_{sid}")
                v_p3_x = st.number_input("Point 3 X (Video)", value=590.0, step=10.0, key=f"v_p3_x_{sid}")
                v_p3_y = st.number_input("Point 3 Y (Video)", value=310.0, step=10.0, key=f"v_p3_y_{sid}")
                v_p4_x = st.number_input("Point 4 X (Video)", value=50.0, step=10.0, key=f"v_p4_x_{sid}")
                v_p4_y = st.number_input("Point 4 Y (Video)", value=310.0, step=10.0, key=f"v_p4_y_{sid}")

            with c_pts2:
                st.markdown("**Floor plan 4 matching corners (pixels)**")
                f_p1_x = st.number_input("Point 1 X (Floor)", value=100.0, step=10.0, key=f"f_p1_x_{sid}")
                f_p1_y = st.number_input("Point 1 Y (Floor)", value=100.0, step=10.0, key=f"f_p1_y_{sid}")
                f_p2_x = st.number_input("Point 2 X (Floor)", value=300.0, step=10.0, key=f"f_p2_x_{sid}")
                f_p2_y = st.number_input("Point 2 Y (Floor)", value=100.0, step=10.0, key=f"f_p2_y_{sid}")
                f_p3_x = st.number_input("Point 3 X (Floor)", value=300.0, step=10.0, key=f"f_p3_x_{sid}")
                f_p3_y = st.number_input("Point 3 Y (Floor)", value=300.0, step=10.0, key=f"f_p3_y_{sid}")
                f_p4_x = st.number_input("Point 4 X (Floor)", value=100.0, step=10.0, key=f"f_p4_x_{sid}")
                f_p4_y = st.number_input("Point 4 Y (Floor)", value=300.0, step=10.0, key=f"f_p4_y_{sid}")

            scale_m = st.number_input("Scale: Pixels per metre", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
            cal_provenance = st.text_input("Calibration notes / provenance", value="Investigator 4-point homography calibration")

            if st.button("Save & apply calibration", type="primary"):
                v_pts = [[v_p1_x, v_p1_y], [v_p2_x, v_p2_y], [v_p3_x, v_p3_y], [v_p4_x, v_p4_y]]
                f_pts = [[f_p1_x, f_p1_y], [f_p2_x, f_p2_y], [f_p3_x, f_p3_y], [f_p4_x, f_p4_y]]
                try:
                    service.add_floor_calibration(
                        cid, actor,
                        source_id=sid,
                        camera_id=cam_id,
                        video_points=v_pts,
                        floor_points=f_pts,
                        floor_plan_asset=floor_asset,
                        pixels_per_metre=scale_m,
                        provenance=cal_provenance
                    )
                    st.success("Calibration saved and applied to case.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Calibration failed: {exc}")
        else:
            st.info("Ingest recordings into this case first.")


def rules(service, case, actor):
    from .rules import validate_rule
    repo, cid = service.repo, case["id"]
    st.caption("Offline flags from sampled trajectories. Rules do not infer intent or criminality.")
    kind = st.selectbox("Rule type", ["dwell", "zone", "line", "crowd", "count", "speed", "wrong_direction", "unattended_object"])
    with st.form("rule"):
        label = st.text_input("Rule name")
        mapping = {c["id"]: c["label"] for c in repo.list(cid, "camera")}
        camera = st.selectbox("Apply to camera", [None] + list(mapping), format_func=lambda c: mapping.get(c, "All cameras"))
        severity = st.selectbox("Severity", ["low", "medium", "high", "critical"], index=1)
        params = {}
        if kind in {"crowd", "count"}:
            params["threshold"] = st.number_input("Distinct sampled person tracklets in one frame", min_value=1, max_value=64, value=4 if kind == "crowd" else 1)
            if kind == "crowd":
                crowd_box_text = st.text_input("Optional normalized crowd rectangle", placeholder="[0.1,0.1,0.9,0.9]")
        elif kind == "unattended_object":
            params["seconds"] = st.number_input("Stationary bag without nearby person (seconds)", min_value=1., max_value=3600., value=20.)
            params["distance_norm"] = st.slider("Nearby distance (fraction of image diagonal)", .01, 1., .15, .01)
            st.caption("Requires Bag detection during processing. Missing person detections can trigger false candidates.")
        elif kind == "dwell":
            params["seconds"] = st.number_input("Stationary visibility threshold (s)", min_value=1., value=30.)
        elif kind == "zone":
            params["box"] = [c.number_input(name, 0., 1., default) for c, name, default in zip(st.columns(4), ["Left", "Top", "Right", "Bottom"], [.2, .2, .8, .8])]
            polygon_text = st.text_input("Optional polygon points as normalized JSON", placeholder="[[0.1,0.2],[0.8,0.2],[0.5,0.9]]")
        elif kind == "speed":
            params["pixels_per_second"] = st.number_input("Image speed threshold (pixels per second)", min_value=.1, value=100.)
            st.caption("Image-plane speed depends on camera perspective; it is not physical speed.")
        elif kind == "wrong_direction":
            params["axis"] = st.selectbox("Image axis", ["x", "y"])
            params["forbidden_direction"] = st.selectbox("Flag movement", ["positive", "negative"], format_func=lambda v: "Right / down" if v == "positive" else "Left / up")
        else:
            params["axis"] = st.selectbox("Axis", ["x", "y"])
            params["position"] = st.slider("Line position", .01, .99, .5)
        if st.form_submit_button("Save rule"):
            if kind == "zone" and polygon_text.strip():
                try:
                    params["polygon"] = json.loads(polygon_text)
                except json.JSONDecodeError as exc:
                    raise ValueError("Enter polygon points as a JSON list of [x,y] pairs.") from exc
            if kind == "crowd" and crowd_box_text.strip():
                try:
                    params["zone_box"] = json.loads(crowd_box_text)
                except json.JSONDecodeError as exc:
                    raise ValueError("Enter crowd rectangle as [left,top,right,bottom].") from exc
            record = {"case_id": cid, "type": kind, "label": label or kind, "camera_id": camera, "severity": severity, **params}
            validate_rule(record)
            repo.put("rule", record, actor)
            st.rerun()
    configured_rules = repo.list(cid, "rule")
    rule_states = {item["rule_id"]: item for item in repo.list(cid, "rule_state")}
    if configured_rules:
        with st.expander("Manage saved rules"):
            st.dataframe([{"Rule": r.get("label", r["type"]), "Type": r["type"],
                           "Severity": r.get("severity", "medium"),
                           "Enabled": rule_states.get(r["id"], {}).get("enabled", True)} for r in configured_rules],
                         hide_index=True, use_container_width=True)
            with st.form("rule_status"):
                selected_rule = st.selectbox("Select rule", configured_rules, format_func=lambda r: r.get("label", r["type"]) + " · " + r["id"][-6:])
                enabled = st.selectbox("Rule status", ["Enabled", "Disabled"],
                                       index=0 if rule_states.get(selected_rule["id"], {}).get("enabled", True) else 1)
                if st.form_submit_button("Save rule status"):
                    repo.put("rule_state", {"case_id": cid, "rule_id": selected_rule["id"], "enabled": enabled == "Enabled", "actor": actor}, actor)
                    st.rerun()
    if st.button("Evaluate batch rules", type="primary"):
        service.run_rules(cid, actor)
        st.rerun()
    events = repo.list(cid, "event")
    latest = {r["event_id"]: r for r in repo.list(cid, "event_review")}
    if not events:
        st.info("No rule events recorded.")
        return
    critical = sum(e.get("severity") == "critical" and e["id"] not in latest for e in events)
    if critical:
        st.markdown(f'<div role="alert" class="critical-alert"><strong>Critical</strong> · {critical} '
                    'unreviewed rule events</div>', unsafe_allow_html=True)
    st.dataframe([{"Event": e["id"], "Method": e["method"], "Severity": e.get("severity", "medium"), "Track": e["track_id"], "Start": e["start_time"], "End": e["end_time"], "Review": latest.get(e["id"], {}).get("decision", "pending")} for e in events], hide_index=True, use_container_width=True)
    with st.form("event_review"):
        eid = st.selectbox("Event to review", [e["id"] for e in events])
        decision = st.selectbox("Event decision", ["acknowledged", "false positive", "escalated", "dismissed", "deferred"])
        reason = st.text_input("Event review reason")
        if st.form_submit_button("Record event review"):
            if not reason.strip():
                raise ValueError("A reason is required.")
            repo.put("event_review", {"case_id": cid, "event_id": eid, "decision": decision, "reason": reason, "actor": actor}, actor)
            st.rerun()


@st.cache_resource
def _live_model():
    from ultralytics import YOLO
    return YOLO("yolo11n.pt")


def live_respond(service, case, actor):
    from .live import LiveSession
    st.write("Monitor a local webcam while this page is open. Saved zone, line, crowd and count rules generate on-screen alerts.")
    st.caption("Local camera only · sampled at the dashboard refresh rate · alert history stays in this browser session. No background service or external notifications.")
    cid = case["id"]
    camera = st.selectbox("Local camera", [0, 1])
    cameras = {c["id"]: c["label"] for c in service.repo.list(cid, "camera")}
    linked_camera = st.selectbox("Treat this webcam as", [None] + list(cameras), format_func=lambda c: cameras.get(c, "Unregistered camera"))
    rule_states = {item["rule_id"]: item for item in service.repo.list(cid, "rule_state")}
    active_rules = [r for r in service.repo.list(cid, "rule") if r["type"] in {"zone", "line", "crowd", "count"}
                    and rule_states.get(r["id"], {}).get("enabled", True)
                    and (not r.get("camera_id") or r["camera_id"] == linked_camera)]
    st.caption(f"{len(active_rules)} compatible rules configured in Rules & events.")
    key = "live_" + cid
    session = st.session_state.get(key)
    if session is None or session.closed:
        if st.button("Start local monitoring", type="primary"):
            st.session_state[key] = LiveSession(camera)
            st.rerun()
        return
    if st.button("Stop monitoring"):
        session.close()
        st.rerun()

    @st.fragment(run_every=1)
    def monitor():
        if session.closed:
            return
        try:
            image, count, alerts = session.step(_live_model(), active_rules)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            session.close()
            return
        st.image(image, caption=f"Camera {session.camera_index} · {count} detected people in latest frame", use_container_width=True)
        for alert in alerts:
            st.toast(f"{alert['rule']}: {alert['count']} detected people", icon="🔔")
        if session.alerts and session.alerts[-1].get("severity") == "critical":
            st.markdown(f'<div role="alert" class="critical-alert"><strong>Critical</strong> · '
                        f'{escape(session.alerts[-1]["rule"])}</div>', unsafe_allow_html=True)
        if session.alerts:
            st.subheader("Live alert history")
            st.dataframe(list(reversed(session.alerts)), hide_index=True, use_container_width=True)
            st.download_button("Download alert history", pd.DataFrame(session.alerts).to_csv(index=False),
                               f"fedsight-live-alerts-{cid[-8:]}.csv", "text/csv")
    monitor()


def analytics(service, case, actor):
    from .analytics import activity_bins, common_paths, dwell_summary
    from .rules import motion_heatmap
    repo, cid = service.repo, case["id"]
    tracks, proposals = service.active_tracks(cid), repo.active_proposals(cid)
    sources = {s["id"]: s for s in repo.list(cid, "source")}
    latest = repo.latest_reviews(cid)
    counts = pd.Series([latest.get(p["id"], {}).get("decision", "pending") for p in proposals], dtype=str).value_counts()
    a, b = st.columns(2)
    if len(counts):
        plot(px.pie(names=counts.index, values=counts.values, hole=.65, title="Human review coverage", color_discrete_sequence=palette()), container=a, use_container_width=True)
    if tracks:
        plot(px.histogram(x=[t.get("embedding_meta", {}).get("mean_quality", 0) for t in tracks], nbins=10, title="Tracklet quality", color_discrete_sequence=palette()), container=b, use_container_width=True)
        st.subheader("Research dashboard")
        key_prefix = "analytics_" + cid + "_"
        presets = repo.list(cid, "dashboard_preset")
        if presets:
            preset = st.selectbox("Saved dashboard", presets, format_func=lambda p: p["label"])
            if st.button("Load dashboard"):
                st.session_state[key_prefix + "sources"] = [s for s in preset["sources"] if s in sources]
                st.session_state[key_prefix + "category"] = preset["category"]
                st.session_state[key_prefix + "interval"] = preset["interval"]
                st.rerun()
        c1, c2, c3 = st.columns(3)
        chosen_sources = c1.multiselect("Sources", list(sources), default=list(sources), format_func=lambda x: sources[x]["name"], key=key_prefix + "sources")
        category = c2.selectbox("Category", ["person", "vehicle", "all"], key=key_prefix + "category")
        interval = c3.selectbox("Activity interval", [10, 30, 60, 120], index=1, format_func=lambda x: f"{x} seconds", key=key_prefix + "interval")
        name = st.text_input("Dashboard name", max_chars=80)
        if st.button("Save dashboard"):
            if not name.strip():
                raise ValueError("Dashboard name is required.")
            repo.put("dashboard_preset", {"case_id": cid, "label": name.strip(), "sources": chosen_sources,
                                          "category": category, "interval": interval}, actor)
            st.rerun()
        filtered = [t for t in tracks if t["source_id"] in chosen_sources and (category == "all" or t.get("object_class", "person") == category)]
        st.metric("Indexed tracklets", len(filtered))
        activity = activity_bins(filtered, bin_seconds=interval)
        if activity:
            frame = pd.DataFrame(activity)
            frame["Source"] = frame["source_id"].map(lambda x: sources[x]["name"])
            plot(px.line(frame, x="start_s", y="visible_tracklets", color="Source", markers=True,
                                    labels={"start_s": "Clip-relative time (s)", "visible_tracklets": "Visible sampled tracklets"},
                                    color_discrete_sequence=palette(), title="Activity trend by source"), use_container_width=True)
        dwell = dwell_summary(filtered)
        if dwell:
            st.subheader("Longest stationary visibility")
            st.dataframe(pd.DataFrame(dwell[:20]).assign(source=lambda f: f["source_id"].map(lambda x: sources[x]["name"])), hide_index=True, use_container_width=True)
        paths = common_paths(filtered)
        if paths:
            st.subheader("Common image-plane paths")
            st.dataframe(pd.DataFrame(paths[:20]).assign(source=lambda f: f["source_id"].map(lambda x: sources[x]["name"])), hide_index=True, use_container_width=True)
        st.subheader("Sampled image-plane motion density")
        if chosen_sources:
            sid = st.selectbox("Density source", chosen_sources, format_func=lambda x: sources[x]["name"])
            selected = [t for t in filtered if t["source_id"] == sid]
            if selected:
                plot(px.imshow(motion_heatmap(filtered, sid, selected[0]["frame_size"]), color_continuous_scale=[tokens()["card"], tokens()["accent"]], labels=dict(x="Image X bin", y="Image Y bin", color="Samples")), use_container_width=True)
        st.caption("Activity and paths use sampled tracklets, so tracking breaks can inflate counts. Dwell is stationary visible duration, not continuous occupancy. Camera planes are not merged.")
    st.caption("Tracklet counts can include fragmentation. Accuracy is reported only after labelled evaluation.")


def evaluation(service, case, actor):
    from .evaluation import synthetic_dataset, train_compare
    repo, cid = service.repo, case["id"]
    st.write("Compare logistic regression, gradient boosting and an MLP with identity-, sequence- and case-disjoint training, calibration, validation and test sets.")
    st.caption("Keep a separate final benchmark. Repeated test-set inspection can bias development.")
    st.download_button("Download dataset format (synthetic example)", canonical(synthetic_dataset()), "fedsight-pair-format.json", "application/json")
    uploaded = st.file_uploader("Labelled pair-feature JSON", type=["json"])
    synthetic = st.checkbox("Run a synthetic software-check experiment", value=case["demo"])
    if st.button("Train, calibrate and evaluate", type="primary", disabled=not synthetic and uploaded is None):
        if uploaded and uploaded.size > 10 * 1024 * 1024:
            raise ValueError("Evaluation input is limited to 10 MB.")
        rows = synthetic_dataset() if synthetic else json.loads(uploaded.getvalue())
        with st.spinner("Checking splits and comparing models…"):
            artifact, report = train_compare(rows)
        repo.put("experiment", {"case_id": cid, "artifact": artifact, "report": report, "actor": actor}, actor)
        st.rerun()
    experiments = {e["id"]: e for e in repo.list(cid, "experiment")}
    if experiments:
        experiment = experiments[st.selectbox("Saved experiment", list(experiments))]
        report, artifact = experiment["report"], experiment["artifact"]
        if report["synthetic"]:
            st.warning("Synthetic feature experiment. These scores test software mechanics, not real person matching.")
        metrics = report["held_out_test"]
        for col, key, label in zip(st.columns(4), ["precision", "recall", "brier", "ece"], ["Test precision", "Test recall", "Brier score", "Calibration error"]):
            col.metric(label, f"{metrics[key]:.3f}")
        st.caption(f"Selected model: {report['selected_model']} · Frozen threshold: {artifact['threshold']:.4f}")
        st.dataframe([{"Model": c["model"], "Threshold": c["threshold"], "Validation F1": c["validation"]["f1"], "Validation PR-AUC": c["validation"]["pr_auc"], "Validation Brier": c["validation"]["brier"]} for c in report["comparisons"]], hide_index=True)
        bins = metrics["reliability_bins"]
        fig = go.Figure([go.Scatter(x=[b["mean_probability"] for b in bins], y=[b["observed_fraction"] for b in bins], mode="lines+markers", name="Observed"), go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash"))])
        fig.update_layout(title="Held-out reliability", xaxis_title="Predicted probability", yaxis_title="Observed fraction")
        plot(fig, use_container_width=True)
        with st.expander("Confusion counts, splits and failure examples"):
            st.json(report)
        st.download_button("Download experiment and numerical model", canonical(experiment), "fedsight-experiment.json", "application/json")
    with st.expander("Federated research sandbox"):
        st.caption("In-process logistic FedAvg/FedProx simulation. No secure aggregation or differential privacy. Flower networking remains an integration task.")
        mu = st.slider("Proximal penalty μ (zero = FedAvg)", 0., 1., .01, .01)
        if st.button("Run three-client synthetic simulation"):
            from .research import federated_simulation
            from .pair_features import FEATURE_NAMES
            rows = [r for r in synthetic_dataset() if r["split"] == "train"]
            clients = [([[r["features"][k] for k in FEATURE_NAMES] for r in rows[i::3]], [r["label"] for r in rows[i::3]]) for i in range(3)]
            output = federated_simulation(clients, mu=mu)
            plot(px.line(pd.DataFrame(output["rounds"]).set_index("round"), color_discrete_sequence=palette()), use_container_width=True)
            st.json({k: v for k, v in output.items() if k not in {"weights", "rounds"}})


def integrity(service, case, actor):
    repo, cid = service.repo, case["id"]
    result = repo.verify()
    for col, label, value in zip(st.columns(3), ["Audit chain", "Encryption", "Audit algorithm"], ["Verified" if result["ok"] else "FAILED", "AES-256-GCM", "HMAC-SHA3-256"]):
        col.metric(label, value)
    st.caption("Reviewer names are local attribution, not authenticated identities. Set FEDSIGHT_KEY_DIR to protected storage for stronger key custody.")
    if not result["ok"]:
        st.error(str(result))
    if st.button("Verify every original and derived asset", type="primary"):
        st.dataframe(service.verify_assets(cid, actor), hide_index=True, use_container_width=True)
    st.subheader("External audit checkpoint")
    st.caption("Keep checkpoints outside this store. A retained checkpoint detects audit-tail deletion; a chain alone cannot.")
    if result["ok"]:
        st.download_button("Export signed checkpoint", canonical(repo.checkpoint()), "fedsight-audit-checkpoint.json", "application/json")
    uploaded = st.file_uploader("Previously retained checkpoint", type=["json"])
    if uploaded and st.button("Verify retained checkpoint"):
        if uploaded.size > 4096:
            raise ValueError("Invalid checkpoint size.")
        st.json(repo.verify(json.loads(uploaded.getvalue())))
    st.subheader("Case audit events")
    st.dataframe([e for e in repo.audit_events() if e["case_id"] == cid][::-1][:200], use_container_width=True, hide_index=True)
    with st.expander("Replay-control demonstration"):
        st.caption("Persistent local replay check; not authentication of a remote sender.")
        seq = st.number_input("Sequence", min_value=1, value=1)
        if st.button("Submit demo envelope"):
            repo.accept_message("local-demo-node", cid, seq, f"demo-message-{seq}", actor)
            st.success("Accepted. Resubmitting this sequence will be rejected and audited.")
