"""Preserve a replayable interval and its lineage in a single case transaction."""
from bisect import bisect_right
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2

from ...core import canonical, digest, uid
from .archive import image, jpeg, utc
from .live_pipeline import rendered


def materialize(service, archive, event):
    if event.get("source_type") != "live":
        raise ValueError("Select a recorded live event.")
    t = event["stamp"]
    rows = archive.samples(event["case_id"],event["camera_id"],t-12,t+12,pixels=True,run=event["run"])
    if not rows or not any(abs(row[0]-t)<.001 for row in rows):
        raise ValueError("This event’s footage has expired from the rolling buffer.")
    # Never bridge a stream outage or replay-loop boundary.
    pivot = min(range(len(rows)),key=lambda i: abs(rows[i][0]-t))
    lo = hi = pivot
    while lo and rows[lo][0]-rows[lo-1][0] <= 2:
        lo -= 1
    while hi+1<len(rows) and rows[hi+1][0]-rows[hi][0] <= 2:
        hi += 1
    rows = rows[lo:hi+1]
    if rows[-1][0]-rows[0][0] < 10:
        raise ValueError("A continuous 10-second interval is needed. Allow the camera to record longer or choose a later event.")
    fps = 4
    start = max(rows[0][0],min(t-5,rows[-1][0]-10+1/fps))
    stamps = [r[0] for r in rows]
    executable = shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    tracks, crops, quality, frames = {}, {}, {}, []
    event_row = min(rows,key=lambda r: abs(r[0]-t))
    snapshot = jpeg(rendered(image(event_row[2]),event_row[1],[],t))
    for i in range(10*fps):
        stamp, observed, blob = rows[max(0,bisect_right(stamps,start+i/fps)-1)]
        frame = image(blob)
        h,w = frame.shape[:2]
        for track in observed:
            tid = track["id"]
            _,x,y,bw,bh = track["trajectory"][-1]
            record = tracks.setdefault(tid,dict(attributes=track["attributes"],avg_conf=track["avg_conf"],trajectory=[]))
            record["trajectory"].append([i,x,y,bw,bh])
            score = bw*bh*track["avg_conf"]
            if score>quality.get(tid,0):
                crop = frame[max(0,int(y-bh/2)):min(h,int(y+bh/2)+1), max(0,int(x-bw/2)):min(w,int(x+bw/2)+1)]
                if crop.size:
                    crops[tid], quality[tid] = jpeg(crop), score
        cv2.putText(frame, utc(stamp),(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        frames.append(frame.tobytes())
    with tempfile.TemporaryDirectory(dir=service.work) as folder:
        target = Path(folder)/"clip.mp4"
        cmd = [executable,"-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
               "-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0","-an","-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v","libx264","-preset","ultrafast","-threads","1","-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
        try:
            subprocess.run(cmd,input=b"".join(frames),capture_output=True,check=True,timeout=45)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise ValueError("The local video encoder could not finish this clip. Check FFmpeg and available storage.") from exc
        clip = target.read_bytes()
    return dict(clip=clip,snapshot=snapshot,crops=crops,tracks=tracks,start=start,seconds=t-start,
                fps=fps,frames=len(frames),size=[w+(w%2),h+(h%2)])


def preserve(service,event,bundle,actor,*,case_id=None,title="",assignee="",status="Created"):
    if not case_id and (not title.strip() or not assignee.strip() or status not in {"Created","Under Review"}):
        raise ValueError("Enter a case name, assignee and opening status.")
    repo = service.repo
    event = repo.get(event["case_id"],event["id"],"event")
    cam = repo.get(event["case_id"],event["camera_id"],"camera")
    new = not case_id
    cid = case_id or uid("case")
    sid,jid,camera = uid("source"),uid("job"),uid("camera")
    files = []
    origin = dict(site_id=event["case_id"],live_event_id=event["id"],live_camera_id=cam["id"],
                  simulated=event["simulated"],masked=event["masked"],start_utc=utc(bundle["start"]),
                  event_utc=event["time_utc"],source_type="live")
    try:
        with repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            def put(kind,**fields):
                return repo._put(db,kind,dict(case_id=cid,**fields),actor)
            def asset(blob,role,parent=None):
                a = service.vault.put(blob,cid,uid("asset"))
                files.append(a)
                put("asset",**{k:v for k,v in a.items() if k!="case_id"},id=a["asset_id"],role=role,
                    source_id=sid,parent_id=parent,provenance=origin)
                return a
            if new:
                put("case",id=cid,title=title.strip(),description="Opened from a live rule flag.",owner=assignee.strip(),
                    status=status,demo=False,consent="")
            else:
                case = repo._decode(repo._case_row(db,cid))
                if case["demo"] or case["status"] not in {"Created","Under Review","Failed"}:
                    raise ValueError("Choose an open, editable case.")
                for row in db.execute("SELECT * FROM records WHERE case_id=? AND kind='source'",(cid,)):
                    existing = repo._decode(row)
                    if existing.get("provenance",{}).get("live_event_id")==event["id"]:
                        return cid,existing["id"],existing["event_seconds"]
                if db.execute("SELECT count(*) FROM records WHERE case_id=? AND kind='source'",(cid,)).fetchone()[0]>=32:
                    raise ValueError("The case already has 32 sources.")
            put("camera",id=camera,label=cam["label"],site=cam.get("site",""),zone=cam.get("zone",""),x=cam.get("x",0),y=cam.get("y",0))
            primary = asset(bundle["clip"],"live_clip")
            still = asset(bundle["snapshot"],"event_snapshot",primary["asset_id"])
            put("source",id=sid,asset=primary,sha256=primary["sha256"],name=cam["label"]+" live event.mp4",suffix=".mp4",
                camera_id=camera,demo=False,time_anchor=dict(original_timestamp=origin["start_utc"],method="workstation capture clock"),
                provenance=origin,event_seconds=bundle["seconds"])
            meta = dict(fps=bundle["fps"],total_frames=bundle["frames"],frame_size=bundle["size"],processed_frames=bundle["frames"],incomplete=False)
            put("job",id=jid,source_id=sid,status="completed",meta=meta,source_type="live",model_manifest={"detector":"YOLO11n","tracker":"MotionIoU"})
            linked = {}
            for tid,r in bundle["tracks"].items():
                points=r["trajectory"]
                first,last=points[0][0],points[-1][0]
                assigned=linked[tid]=uid("track")
                preview=asset(bundle["crops"][tid],"preview",primary["asset_id"]) if tid in bundle["crops"] else None
                put("track",**r,id=assigned,track_id=assigned,tracker_id=tid,source_id=sid,source_key=sid,camera_id=camera,
                    job_id=jid,object_class="person",first_frame=first,last_frame=last,frame_count=len(points),frame_size=bundle["size"],
                    start_time=first/bundle["fps"],end_time=last/bundle["fps"],duration_s=(last-first)/bundle["fps"],
                    preview_asset=preview,quality_report={},motion={},provenance=origin)
            put("event",source_id=sid,source_type="live",track_id=linked.get(event.get("track_id")),camera_id=camera,
                rule_id=event["rule_id"],method=event["method"],severity=event["severity"],start_time=bundle["seconds"],
                end_time=bundle["seconds"],explanation=event["rule"],snapshot=still,provenance=origin)
            repo._append(db,cid,sid,actor,"live.open_as_case",digest(canonical(origin)))
    except Exception:
        for a in files:
            service.vault.remove(cid,a["asset_id"],a["sha256"])
        raise
    return cid,sid,bundle["seconds"]
