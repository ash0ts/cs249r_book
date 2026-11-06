"""Conservative shared reservations; unknown charges never become zero."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


class Budget:
    """One database for the entire campaign, including notebook and retry allowances.

    Reservations remain charged at their maximum until a separate billing audit.
    A client timeout does not establish that remote work stopped.
    """
    def __init__(self, path: Path, *, ceiling_usd: float):
        if not math.isfinite(ceiling_usd) or not 0 < ceiling_usd <= 100:
            raise ValueError("Ceiling must be positive and at most $100")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, ceiling REAL)")
            db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (ceiling_usd,))
            if db.execute("SELECT ceiling FROM settings").fetchone()[0] != ceiling_usd:
                raise ValueError("Cannot reset an existing campaign ceiling")
            db.execute("CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, maximum REAL, basis TEXT, result TEXT)")
        path.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, action_id: str, maximum_usd: float, basis: str) -> bool:
        if not action_id or not basis or not math.isfinite(maximum_usd) or maximum_usd <= 0:
            raise ValueError("A documented, positive cost bound is required")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT maximum,basis FROM actions WHERE id=?", (action_id,)).fetchone()
            if existing:
                if existing != (maximum_usd, basis):
                    raise ValueError("Action identifier reused with a different reservation")
                return False
            used = db.execute("SELECT COALESCE(SUM(maximum),0) FROM actions").fetchone()[0]
            ceiling = db.execute("SELECT ceiling FROM settings").fetchone()[0]
            if used + maximum_usd > ceiling:
                raise ValueError("Campaign reservation would exceed the shared ceiling")
            db.execute("INSERT INTO actions VALUES (?,?,?,NULL)", (action_id, maximum_usd, basis))
        return True

    def complete(self, action_id: str, result: dict):
        with self.connect() as db:
            changed = db.execute("UPDATE actions SET result=? WHERE id=? AND result IS NULL",
                                 (json.dumps(result, sort_keys=True), action_id)).rowcount
            if changed != 1:
                raise ValueError("Unknown or already completed action")

    def result(self, action_id: str):
        with self.connect() as db:
            row = db.execute("SELECT result FROM actions WHERE id=?", (action_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def summary(self):
        with self.connect() as db:
            return {"ceiling_usd": db.execute("SELECT ceiling FROM settings").fetchone()[0],
                    "reserved_usd": db.execute("SELECT COALESCE(SUM(maximum),0) FROM actions").fetchone()[0],
                    "actual_billed_usd": None}
