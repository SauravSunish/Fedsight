from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timezone


def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(path: Path) -> None:
    with _connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                pair_key TEXT NOT NULL,
                track_a TEXT NOT NULL,
                track_b TEXT NOT NULL,
                similarity REAL NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(case_id, pair_key)
            )
        """)
        conn.commit()


def save_decision(path: Path, *, case_id: str, pair_key: str, track_a: str,
                  track_b: str, similarity: float, decision: str) -> None:
    allowed = {"Same appearance / accept link", "Different appearance / reject link"}
    if decision not in allowed:
        raise ValueError("Unsupported review decision")
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            INSERT INTO review_decisions
            (case_id, pair_key, track_a, track_b, similarity, decision, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id, pair_key) DO UPDATE SET
              similarity=excluded.similarity,
              decision=excluded.decision,
              created_at=excluded.created_at
        """, (case_id, pair_key, track_a, track_b, float(similarity), decision, timestamp))
        conn.commit()


def load_decisions(path: Path, case_id: str) -> dict[str, str]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT pair_key, decision FROM review_decisions WHERE case_id=?",
            (case_id,),
        ).fetchall()
    return {pair_key: decision for pair_key, decision in rows}
