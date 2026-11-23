"""Bounded encrypted rolling footage, independent of preserved case evidence."""
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone

import cv2
import numpy as np

from ...core import canonical
from ...security import decrypt, encrypt


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="milliseconds")


def jpeg(frame):
    ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode camera frame.")
    return image.tobytes()


def image(blob):
    frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unreadable camera frame.")
    return frame


class Archive:
    def __init__(self, service):
        self.path = service.root / "live-buffer.sqlite3"
        self.key = service.keys.encryption
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS samples (
                  site TEXT, camera TEXT, ts REAL, run TEXT, pixels BLOB, meta BLOB,
                  PRIMARY KEY(site,camera,ts));
                CREATE INDEX IF NOT EXISTS samples_time ON samples(ts);
                CREATE TABLE IF NOT EXISTS minutes (
                  site TEXT,camera TEXT,minute INTEGER,count_sum INTEGER,n INTEGER,
                  heat BLOB, PRIMARY KEY(site,camera,minute));
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            yield db

    def append(self, site, camera, stamp, run, frame, tracks):
        ctx = dict(site=site, camera=camera, ts=stamp, run=run)
        minute = int(stamp // 60)
        h, w = frame.shape[:2]
        heat = np.zeros((16, 16), dtype=np.int32)
        for track in tracks:
            _, x, y, _, _ = track["trajectory"][-1]
            heat[min(15, max(0, int(y/h*16))), min(15, max(0, int(x/w*16)))] += 1
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?)",
                       (site, camera, stamp, run, encrypt(jpeg(frame), self.key, {**ctx, "field":"pixels"}),
                        encrypt(canonical(tracks), self.key, {**ctx, "field":"tracks"})))
            row = db.execute("SELECT heat FROM minutes WHERE site=? AND camera=? AND minute=?",
                             (site,camera,minute)).fetchone()
            if row:
                heat += np.frombuffer(row[0], dtype=np.int32).reshape(16,16)
            db.execute("""INSERT INTO minutes VALUES (?,?,?,?,?,?) ON CONFLICT(site,camera,minute)
                       DO UPDATE SET count_sum=count_sum+excluded.count_sum,n=n+1,heat=excluded.heat""",
                       (site,camera,minute,len(tracks),1,heat.tobytes()))

    def samples(self, site, camera, start, end, *, pixels=False, run=None):
        field = "meta,pixels" if pixels else "meta,NULL"
        with self.connect() as db:
            rows = db.execute(f"SELECT ts,run,{field} FROM samples WHERE site=? AND camera=? AND ts BETWEEN ? AND ? AND (? IS NULL OR run=?) ORDER BY ts",
                              (site,camera,start,end,run,run)).fetchall()
        result = []
        for ts, session, meta, frame in rows:
            ctx = dict(site=site,camera=camera,ts=ts,run=session)
            result.append((ts, json.loads(decrypt(meta,self.key,{**ctx,"field":"tracks"})),
                           decrypt(frame,self.key,{**ctx,"field":"pixels"}) if pixels else None))
        return result

    def usage(self, site):
        with self.connect() as db:
            return db.execute("SELECT coalesce(sum(length(pixels)+length(meta)),0) FROM samples WHERE site=?", (site,)).fetchone()[0]

    def cleanup(self, site, minutes, byte_cap, current=None):
        current = time.time() if current is None else current
        with self.connect() as db:
            db.execute("DELETE FROM samples WHERE site=? AND ts<?", (site,current-minutes*60))
            db.execute("""DELETE FROM samples WHERE rowid IN (
                          SELECT rid FROM (SELECT rowid AS rid, sum(length(pixels)+length(meta)) OVER
                          (ORDER BY ts DESC,rowid DESC) AS used FROM samples WHERE site=?) WHERE used>?)""",
                       (site,byte_cap))
            db.execute("DELETE FROM minutes WHERE minute<?", (int(current//60)-1440,))

    def statistics(self, site, since):
        with self.connect() as db:
            return db.execute("SELECT camera,minute,count_sum*1.0/n,heat FROM minutes WHERE site=? AND minute>=? ORDER BY minute",
                              (site,int(since//60))).fetchall()
