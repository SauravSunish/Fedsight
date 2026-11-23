"""Case-scoped encrypted records. Every mutation and review has an audit event."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .core import canonical, digest, now, uid
from .security import Keyring, IntegrityError, encrypt, decrypt, audit_mac, verify_events

CASE_TRANSITIONS = {
    "Created": {"Processing", "Archived"}, "Processing": {"Under Review", "Failed"},
    "Failed": {"Processing", "Archived"}, "Under Review": {"Reviewed", "Processing", "Archived"},
    "Reviewed": {"Closed", "Under Review", "Processing"}, "Closed": {"Under Review", "Archived"}, "Archived": {"Under Review"},
}


class Repository:
    def __init__(self, path: Path, keys: Keyring):
        self.path, self.keys = Path(path), keys
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records (
                  id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT NOT NULL,
                  payload BLOB NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS case_kind ON records(case_id,kind);
                CREATE TABLE IF NOT EXISTS audit (
                  seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
                  case_id TEXT NOT NULL, record_id TEXT NOT NULL, evidence_hash TEXT NOT NULL,
                  previous TEXT NOT NULL, mac TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS replay (
                  node_id TEXT NOT NULL, case_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                  message_id TEXT NOT NULL, UNIQUE(node_id,case_id,sequence), UNIQUE(node_id,case_id,message_id));
                CREATE TABLE IF NOT EXISTS pending_asset_removal (
                  case_id TEXT NOT NULL, asset_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                  PRIMARY KEY (case_id, asset_id));
            ''')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _append(self, conn, case_id, record_id, actor, action, evidence_hash):
        if not str(actor).strip():
            raise ValueError("An actor is required.")
        last = conn.execute("SELECT seq,mac FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        event = {"seq": last["seq"] + 1 if last else 1, "at": now(), "actor": actor,
                 "action": action, "case_id": case_id, "record_id": record_id,
                 "evidence_hash": evidence_hash, "previous": last["mac"] if last else "0" * 64}
        event["mac"] = audit_mac(self.keys.audit, event)
        conn.execute("INSERT INTO audit VALUES (:seq,:at,:actor,:action,:case_id,:record_id,:evidence_hash,:previous,:mac)", event)

    def put(self, kind: str, record: dict, actor: str, *, replace: bool = False) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._put(conn, kind, record, actor, replace=replace)

    def _case_row(self, conn, case_id):
        row = conn.execute("SELECT * FROM records WHERE id=? AND case_id=? AND kind='case'",
                           (case_id, case_id)).fetchone()
        if row is None:
            raise ValueError("This case is no longer available.")
        return row

    def _put(self, conn, kind, record, actor, *, replace=False):
        value = dict(record)
        value.setdefault("id", uid(kind))
        value.setdefault("created_at", now())
        if kind == "live_site":
            if value["id"] != value["case_id"] or not value["id"].startswith("site_"):
                raise ValueError("Invalid live site scope.")
        elif kind != "case":
            if value["case_id"].startswith("site_"):
                if not conn.execute("SELECT 1 FROM records WHERE id=? AND kind='live_site'",
                                    (value["case_id"],)).fetchone():
                    raise ValueError("Unknown live site.")
            else:
                self._case_row(conn, value["case_id"])
        raw = canonical(value)
        context = {"id": value["id"], "case_id": value["case_id"], "kind": kind}
        payload = encrypt(raw, self.keys.encryption, context)
        existing = conn.execute("SELECT case_id,kind FROM records WHERE id=?", (value["id"],)).fetchone()
        if existing and (not replace or existing["case_id"] != value["case_id"] or existing["kind"] != kind):
            raise ValueError("Immutable or differently scoped record already exists.")
        if replace and (not existing or kind not in {"case", "job"}):
            raise ValueError("Only an existing case or job may be updated.")
        if kind == "case" and not existing and conn.execute(
            "SELECT 1 FROM audit WHERE case_id=? AND action='case.deleted' LIMIT 1", (value["case_id"],)
        ).fetchone():
            raise ValueError("This case was permanently deleted.")
        conn.execute("INSERT INTO records VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                     (value["id"], value["case_id"], kind, payload, value["created_at"]))
        self._append(conn, value["case_id"], value["id"], actor, f"{kind}.{'updated' if existing else 'created'}", digest(raw))
        return json.loads(raw)

    def _decode(self, row):
        return json.loads(decrypt(row["payload"], self.keys.encryption, {k: row[k] for k in ("id", "case_id", "kind")}))

    def get(self, case_id: str, record_id: str, kind: str | None = None) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM records WHERE case_id=? AND id=?", (case_id, record_id)).fetchone()
        if row is None or (kind and row["kind"] != kind):
            raise ValueError("Record is unavailable in this case.")
        return self._decode(row)

    def list(self, case_id: str | None, kind: str) -> list[dict]:
        if case_id is None and kind not in {"case", "live_site"}:
            raise ValueError("A case scope is required.")
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM records WHERE kind=? AND (? IS NULL OR case_id=?) ORDER BY created_at,id",
                                (kind, case_id, case_id)).fetchall()
        return [self._decode(row) for row in rows]

    def create_case(self, title: str, description: str, actor: str, *, demo=False, consent=""):
        if not title.strip():
            raise ValueError("Case title is required.")
        case_id = uid("case")
        return self.put("case", {"id": case_id, "case_id": case_id, "title": title.strip(), "description": description,
                                "owner": actor, "status": "Created", "demo": demo, "consent": consent}, actor)

    def edit_case(self, case_id, title, description, consent, actor):
        if not title.strip():
            raise ValueError("Case title is required.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            case = self._decode(self._case_row(conn, case_id))
            case.update(title=title.strip(), description=description, consent=consent)
            return self._put(conn, "case", case, actor, replace=True)

    def delete_case(self, case_id, actor):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._case_row(conn, case_id)
            for row in conn.execute("SELECT * FROM records WHERE case_id=? AND kind='asset'", (case_id,)):
                asset = self._decode(row)
                conn.execute("INSERT INTO pending_asset_removal VALUES (?,?,?)",
                             (case_id, asset["asset_id"], asset["sha256"]))
            conn.execute("DELETE FROM replay WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM records WHERE case_id=?", (case_id,))
            self._append(conn, case_id, case_id, actor, "case.deleted", "")

    def storage_info(self, case_id=None):
        with self.connect() as conn:
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            return {
                "database_bytes": page_size * conn.execute("PRAGMA page_count").fetchone()[0],
                "unused_bytes": page_size * conn.execute("PRAGMA freelist_count").fetchone()[0],
                "pending_files": conn.execute("SELECT COUNT(*) FROM pending_asset_removal").fetchone()[0],
                "counts": dict(conn.execute("SELECT kind, COUNT(*) FROM records WHERE case_id=? GROUP BY kind",
                                            (case_id,)).fetchall()),
            }

    def compact(self):
        with self.connect() as conn:
            conn.execute("VACUUM")

    def transition(self, case_id, status, actor):
        case = self.get(case_id, case_id, "case")
        if status not in CASE_TRANSITIONS[case["status"]]:
            raise ValueError(f"Cannot transition from {case['status']} to {status}.")
        if status in {"Reviewed", "Closed"}:
            decisions = self.latest_reviews(case_id)
            proposals = self.active_proposals(case_id)
            if any(decisions.get(p["id"], {}).get("decision") not in {"accept", "reject"} for p in proposals):
                raise ValueError("Resolve pending/deferred proposals before marking the case reviewed.")
        if status == "Closed" and not self.verify()["ok"]:
            raise IntegrityError("Audit verification must pass before closing a case.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._decode(self._case_row(conn, case_id))
            if current["status"] != case["status"]:
                raise ValueError("Case status changed. Refresh and try again.")
            current["status"] = status
            return self._put(conn, "case", current, actor, replace=True)

    def review(self, case_id, proposal_id, actor, decision, reason):
        proposal = self.get(case_id, proposal_id, "proposal")
        case = self.get(case_id, case_id, "case")
        if case["status"] != "Under Review":
            raise ValueError("Open the case for review before recording decisions.")
        if decision not in {"accept", "reject", "defer"} or not reason.strip():
            raise ValueError("Choose accept/reject/defer and provide a reason.")
        return self.put("review", {"case_id": case_id, "proposal_id": proposal_id, "actor": actor,
                                  "decision": decision, "reason": reason.strip(), "run_id": proposal["run_id"]}, actor)

    def latest_reviews(self, case_id):
        return {r["proposal_id"]: r for r in self.list(case_id, "review")}

    def active_proposals(self, case_id):
        runs = self.list(case_id, "run")
        if not runs:
            return []
        run_id = runs[-1]["id"]
        return [p for p in self.list(case_id, "proposal") if p["run_id"] == run_id]

    def audit_events(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM audit ORDER BY seq")]

    def audit(self, case_id, actor, action, record_id="", evidence_hash=""):
        if action == "case.deleted":
            raise ValueError("Use case deletion to record a removal.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._append(conn, case_id, record_id, actor, action, evidence_hash)

    def verify(self, checkpoint=None):
        with self.connect() as conn:
            conn.execute("BEGIN")
            events = [dict(row) for row in conn.execute("SELECT * FROM audit ORDER BY seq")]
            rows = conn.execute("SELECT * FROM records").fetchall()
        result = verify_events(events, self.keys.audit, checkpoint)
        if not result["ok"]:
            return result
        deletions = {e["case_id"]: e["seq"] for e in events if e["action"] == "case.deleted"}
        latest = {e["record_id"]: e for e in events if e["action"].endswith((".created", ".updated"))
                  and e["seq"] > deletions.get(e["case_id"], 0)}
        if {r["id"] for r in rows} != set(latest):
            return {"ok": False, "reason": "Stored records differ from audited record inventory."}
        for row in rows:
            try:
                if digest(canonical(self._decode(row))) != latest[row["id"]]["evidence_hash"]:
                    raise IntegrityError("Record digest mismatch.")
            except (ValueError, KeyError) as exc:
                return {"ok": False, "record_id": row["id"], "reason": str(exc)}
        return result

    def checkpoint(self):
        result = self.verify()
        if not result["ok"]:
            raise IntegrityError("Cannot checkpoint a broken history.")
        checkpoint = {"seq": result["events"], "mac": result["head"]}
        return {**checkpoint, "signature": audit_mac(self.keys.audit, checkpoint)}

    def accept_message(self, node_id, case_id, sequence, message_id, actor):
        if not node_id or not message_id or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("Invalid message envelope.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._case_row(conn, case_id)
            last = conn.execute("SELECT MAX(sequence) FROM replay WHERE node_id=? AND case_id=?", (node_id, case_id)).fetchone()[0]
            duplicate = conn.execute("SELECT 1 FROM replay WHERE node_id=? AND case_id=? AND message_id=?", (node_id, case_id, message_id)).fetchone()
            rejected = duplicate or (last is not None and sequence <= last)
            if not rejected:
                conn.execute("INSERT INTO replay VALUES (?,?,?,?)", (node_id, case_id, sequence, message_id))
            self._append(conn, case_id, "", actor, "replay.rejected" if rejected else "message.accepted",
                         digest(canonical({"node": node_id, "sequence": sequence, "message_id": message_id})))
        if rejected:
            raise IntegrityError("Duplicate or stale message rejected.")
