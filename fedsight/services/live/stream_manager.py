"""Site-scoped streams with independent capture, inference and retention workers."""
from collections import deque
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from urllib.parse import urlsplit
import atexit
import math
import os
import sqlite3
import tempfile
import time

import cv2

from ...core import uid
from .archive import Archive, utc
from .live_pipeline import Detector, Pipeline, protected


SAMPLES = Path(__file__).resolve().parents[3] / "data" / "sample_videos"


def site_record(repo, actor, label="Local site", site_id="site_local"):
    if not label.strip():
        raise ValueError("Enter a site name.")
    with repo.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM records WHERE id=? AND kind='live_site'", (site_id,)).fetchone()
        if row:
            return repo._decode(row)
        return repo._put(db, "live_site", dict(id=site_id,case_id=site_id,label=label.strip()), actor)


def source_for_camera(repo, site, camera_id):
    camera = repo.get(site, camera_id, "camera")
    with repo.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for row in db.execute("SELECT * FROM records WHERE case_id=? AND kind='source'", (site,)):
            source = repo._decode(row)
            if source.get("camera_id") == camera_id:
                return source["id"]
        source = repo._put(db, "source", dict(case_id=site, camera_id=camera_id, name=camera["label"],
                                              source_type="live", time_anchor={}, demo=False), "live-worker")
        return source["id"]


def active_rules(repo, site, *, include_disabled=False):
    states = {s["rule_id"]:s for s in repo.list(site,"rule_state")}
    return [{**r, **{k:v for k,v in states.get(r["id"], {}).items() if k in ("enabled","severity","hours")}}
            for r in repo.list(site,"rule") if include_disabled or states.get(r["id"],r).get("enabled", True)]


def capture_source(spec, service):
    source = spec["source"]
    kind = spec["kind"]
    temporary = None
    if kind == "usb":
        source = int(source)
        if source not in range(8):
            raise ValueError("Use a local camera index from 0 to 7.")
    elif kind == "demo":
        path = (SAMPLES / source).resolve()
        if SAMPLES.resolve() not in path.parents or not path.is_file():
            raise ValueError("Demo recording is missing.")
        source = str(path)
    elif kind == "upload":
        blob = service.read_asset(spec["asset"], "live-worker")
        with tempfile.NamedTemporaryFile(dir=service.work, suffix=Path(source).suffix, delete=False) as out:
            temporary = Path(out.name)
            os.chmod(temporary,0o600)
            out.write(blob)
        source = str(temporary)
    elif kind != "rtsp" or urlsplit(source).scheme.lower() not in ("rtsp","rtsps"):
        raise ValueError("Use an RTSP address, a local camera, or a video.")
    try:
        if kind == "rtsp":
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,
                                                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
        else:
            cap = cv2.VideoCapture(source)
        return cap, temporary
    except Exception:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


class Stream:
    def __init__(self, manager, site, config):
        self.manager, self.site, self.config = manager,site,config
        self.camera = config["camera_id"]
        self.stop_event = Event()
        self.pause_event = Event()
        self.condition = Condition()
        self.queue = deque(maxlen=1)
        self.latest = None
        self.errors = deque(maxlen=8)
        self.status = "Starting"
        self.last_event = None
        self.fps = 0.
        self.last_processed = 0.
        self.lag = 0.
        self.dropped = 0
        self.workers = [Thread(target=f,daemon=True) for f in (self.capture,self.analyze)]

    def pause(self):
        self.pause_event.set()
        with self.condition:
            self.status = "Paused"
            self.condition.notify_all()

    def resume(self):
        self.pause_event.clear()
        with self.condition:
            self.status = "Online"
            self.condition.notify_all()

    def is_paused(self):
        return self.pause_event.is_set()

    def error(self, message):
        with self.condition:
            self.errors.append((utc(time.time()),message))
            self.status = message

    def capture(self):
        retry = .5
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.stop_event.wait(0.5)
                continue
            cap = temp = None
            try:
                cap,temp = self.manager.opener(self.config,self.manager.service)
                if not cap.isOpened():
                    raise OSError("Cannot open source")
                playback = self.config["kind"] in ("demo","upload")
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if math.isfinite(fps) and 1 <= fps <= 240 else 25.
                epoch = uid("capture")
                due = time.monotonic()
                last_sample = 0.
                while not self.stop_event.is_set():
                    if self.pause_event.is_set():
                        self.stop_event.wait(0.5)
                        continue
                    ok,frame = cap.read()
                    if not ok or frame is None:
                        if playback and cap.set(cv2.CAP_PROP_POS_FRAMES,0):
                            ok,frame = cap.read()
                            epoch = uid("capture")
                        if not ok or frame is None:
                            raise OSError("Source interrupted")
                    current = time.monotonic()
                    if current-last_sample >= 1 / self.config.get("sample_fps",3):
                        last_sample = current
                        h,w = frame.shape[:2]
                        if w > 640:
                            frame = cv2.resize(frame, (640, max(2,round(h*640/w))), interpolation=cv2.INTER_AREA)
                        with self.condition:
                            if self.queue:
                                self.dropped += 1
                            self.queue.append((time.time(),epoch,frame))
                            self.condition.notify()
                        retry = .5
                    if playback:
                        due = max(current, due+1/fps)
                        self.stop_event.wait(max(0.,due-time.monotonic()))
            except Exception as exc:
                if not self.pause_event.is_set():
                    self.error(f"Offline · {type(exc).__name__} · reconnecting")
            finally:
                if cap is not None:
                    cap.release()
                if temp:
                    temp.unlink(missing_ok=True)
            if not self.stop_event.wait(retry):
                retry = min(15.,retry*2)

    def analyze(self):
        epoch = None
        rules = []
        next_rules = 0.
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.stop_event.wait(0.5)
                continue
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.stop_event.is_set() or self.pause_event.is_set(),timeout=1)
                if self.stop_event.is_set() or self.pause_event.is_set() or not self.queue:
                    continue
                stamp,session,raw = self.queue.pop()
            try:
                if session != epoch:
                    pipeline = Pipeline(self.manager.detector)
                    epoch = session
                if stamp >= next_rules:
                    rules = [r for r in active_rules(self.manager.service.repo,self.site)
                             if r.get("camera_id") in (None,self.camera)]
                    next_rules = stamp+2
                tracks,hits = pipeline.step(raw,rules,self.camera,stamp)
                if self.stop_event.is_set() or self.pause_event.is_set():
                    return
                frame = protected(raw,self.config.get("masked",False))
                self.manager.archive.append(self.site,self.camera,stamp,pipeline.run,frame,tracks)
                for hit in hits:
                    self.last_event = self.manager.event(self,stamp,pipeline.run,hit,len(tracks))
                current = time.time()
                with self.condition:
                    if self.latest and self.latest[0] < stamp:
                        rate = 1 / (stamp - self.latest[0])
                        self.fps = .8*self.fps + .2*rate if self.fps else rate
                    self.last_processed = current
                    self.lag = max(0,current-stamp)
                    self.latest = (stamp,frame,tracks,rules,pipeline.run)
                    self.status = "Online"
            except Exception as exc:
                if not self.pause_event.is_set():
                    self.error(f"Analysis error · {type(exc).__name__}")
                self.stop_event.wait(2)

    def snapshot(self):
        with self.condition:
            if self.pause_event.is_set():
                status = "Paused"
            else:
                stale = self.last_processed and time.time()-self.last_processed > 8
                status = "Stale" if self.status == "Online" and stale else self.status
            return dict(status=status,fps=round(self.fps,1) if not self.pause_event.is_set() else 0.,
                        lag=round(self.lag,2) if not self.pause_event.is_set() else 0.,queue=len(self.queue),
                        dropped=self.dropped,latest=self.latest,last_event=self.last_event,errors=list(self.errors))


class StreamManager:
    def __init__(self, service, detector=None, opener=capture_source):
        self.service, self.opener = service,opener
        self.detector = detector or Detector()
        self.archive = Archive(service)
        self.streams = {}
        self.revisions = {}
        self.lock = RLock()
        self.bus = deque(maxlen=256)
        self.quit = Event()
        self.retention_error = None
        self.janitor = Thread(target=self.maintenance,daemon=True)
        self.janitor.start()
        atexit.register(self.close)

    def maintenance(self):
        while not self.quit.is_set():
            try:
                for site in self.service.repo.list(None,"live_site"):
                    config = self.service.repo.list(site["id"],"retention")
                    settings = config[-1] if config else {}
                    self.archive.cleanup(site["id"],settings.get("minutes",15),int(settings.get("gb",.25)*1024**3))
                self.retention_error = None
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                self.retention_error = f"Buffer cleanup failed · {type(exc).__name__}"
            self.quit.wait(10)

    def start(self, site, spec):
        key = site,spec["camera_id"]
        with self.lock:
            existing = self.streams.get(key)
            if existing and any(t.is_alive() for t in existing.workers):
                return False
            source_id = source_for_camera(self.service.repo, site, spec["camera_id"])
            worker = self.streams[key] = Stream(self,site,{**spec, "source_id": source_id})
            for t in worker.workers:
                t.start()
            return True

    def stop(self, site, camera):
        with self.lock:
            self.revisions[(site,camera)] = self.revisions.get((site,camera),0) + 1
            stream = self.streams.get((site,camera))
            if stream:
                with stream.condition:
                    stream.stop_event.set()
                    stream.condition.notify_all()
                    stream.status = "Stopping"
                    stream.latest = None

    def restart(self, site, config):
        camera = config["camera_id"]
        with self.lock:
            old = self.streams.get((site,camera))
            self.stop(site,camera)
            revision = self.revisions[(site,camera)]
        def replace():
            if old:
                for worker in old.workers:
                    worker.join()
            with self.lock:
                if self.revisions.get((site,camera)) == revision and not self.quit.is_set():
                    self.start(site,config)
        Thread(target=replace,daemon=True).start()

    def close(self):
        self.quit.set()
        with self.lock:
            for site,camera in self.streams:
                self.stop(site,camera)

    def stop_all(self, site):
        with self.lock:
            for (s, camera) in list(self.streams.keys()):
                if s == site:
                    self.stop(site, camera)

    def pause_all(self, site):
        with self.lock:
            for (s, _), stream in self.streams.items():
                if s == site:
                    stream.pause()

    def resume_all(self, site):
        with self.lock:
            for (s, _), stream in self.streams.items():
                if s == site:
                    stream.resume()

    def is_paused(self, site):
        with self.lock:
            active = [s for (s_site, _), s in self.streams.items() if s_site == site and any(t.is_alive() for t in s.workers)]
            if not active:
                return False
            return all(s.is_paused() for s in active)

    def is_active(self, site):
        with self.lock:
            return any(s_site == site and any(t.is_alive() for t in s.workers) and not s.is_paused()
                       for (s_site, _), s in self.streams.items())

    def snapshot(self,site,camera):
        with self.lock:
            s = self.streams.get((site,camera))
            if s and any(t.is_alive() for t in s.workers):
                return s.snapshot()
            return dict(status="Stopped",fps=0.,lag=0.,queue=0,dropped=0,latest=None,last_event=None,errors=[])

    def source_id(self,site,camera):
        with self.lock:
            return self.streams[(site,camera)].config["source_id"]

    def last_flag(self, site):
        with self.lock:
            latest = next((ts for scope, _, ts in reversed(self.bus) if scope == site), None)
        if latest:
            return latest
        repo = self.service.repo
        with repo.connect() as db:
            row = db.execute("SELECT * FROM records WHERE case_id=? AND kind='event' ORDER BY created_at DESC LIMIT 1",
                             (site,)).fetchone()
            return repo._decode(row)["time_utc"] if row else None

    def event(self, worker, stamp, run, fields, count):
        record = self.service.repo.put("event",dict(case_id=worker.site,source_type="live",source_id=worker.config["source_id"],
                          camera_id=worker.camera,stamp=stamp,time_utc=utc(stamp),run=run,count=count,
                          simulated=worker.config["kind"] in ("demo","upload"),masked=worker.config.get("masked",False),
                          start_time=stamp,end_time=stamp,**fields),"live-worker")
        with self.lock:
            self.bus.append((worker.site,record["id"],record["time_utc"]))
        return record
