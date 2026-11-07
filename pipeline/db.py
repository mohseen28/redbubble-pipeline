"""SQLite persistence. Images are stored as paths only."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.schemas import DesignState, Status, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    niche       TEXT,
    auto_niche  INTEGER NOT NULL DEFAULT 0,
    n           INTEGER NOT NULL,
    dry_run     INTEGER NOT NULL DEFAULT 0,
    prompt_done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS designs (
    design_id   TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    slug        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL,
    state_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_designs_status ON designs(status);
CREATE INDEX IF NOT EXISTS idx_designs_run ON designs(run_id);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id   TEXT,
    ts          TEXT NOT NULL,
    stage       TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT,
    traceback   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_design ON events(design_id);
CREATE TABLE IF NOT EXISTS embeddings (
    design_id   TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL
);
"""


@dataclass(frozen=True)
class RunRow:
    run_id: str
    created_at: str
    niche: str | None
    auto_niche: bool
    n: int
    dry_run: bool
    prompt_done: bool


@dataclass(frozen=True)
class EventRow:
    id: int
    design_id: str | None
    ts: str
    stage: str
    action: str
    detail: str | None
    traceback: str | None


class DB:
    """Thin SQLite wrapper. Every write is committed immediately (resumability)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ runs
    def create_run(self, niche: str | None, auto_niche: bool, n: int, dry_run: bool,
                   now: datetime | None = None) -> str:
        now = now or utcnow()
        prefix = now.strftime("%Y%m%d") + "_run"
        with self._tx() as c:
            count = c.execute("SELECT COUNT(*) FROM runs WHERE run_id LIKE ?", (prefix + "%",)).fetchone()[0]
            run_id = f"{prefix}{count + 1:02d}"
            c.execute(
                "INSERT INTO runs(run_id, created_at, niche, auto_niche, n, dry_run) VALUES (?,?,?,?,?,?)",
                (run_id, now.isoformat(), niche, int(auto_niche), n, int(dry_run)),
            )
        return run_id

    def get_run(self, run_id: str) -> RunRow | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return _run_row(row) if row else None

    def list_runs(self) -> list[RunRow]:
        return [_run_row(r) for r in self._conn.execute("SELECT * FROM runs ORDER BY created_at, run_id")]

    def latest_run(self) -> RunRow | None:
        runs = self.list_runs()
        return runs[-1] if runs else None

    def find_resumable_run(self, niche: str | None, auto_niche: bool, n: int, dry_run: bool) -> RunRow | None:
        """Latest run with identical arguments whose stage 1 is unfinished or has open designs."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE IFNULL(niche,'')=IFNULL(?, '') AND auto_niche=? AND n=? AND dry_run=? "
            "ORDER BY created_at DESC, run_id DESC",
            (niche, int(auto_niche), n, int(dry_run)),
        ).fetchall()
        for row in rows:
            run = _run_row(row)
            if not run.prompt_done:
                return run
            open_count = self._conn.execute(
                "SELECT COUNT(*) FROM designs WHERE run_id=? AND status NOT IN "
                "('exported','rejected','qa_failed','error')",
                (run.run_id,),
            ).fetchone()[0]
            if open_count:
                return run
            return None
        return None

    def mark_prompt_done(self, run_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE runs SET prompt_done=1 WHERE run_id=?", (run_id,))

    # --------------------------------------------------------------- designs
    def insert(self, state: DesignState, run_id: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO designs(design_id, run_id, slug, status, state_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (state.design_id, run_id, state.slug, state.status, state.model_dump_json(),
                 state.created_at.isoformat(), state.updated_at.isoformat()),
            )

    def save(self, state: DesignState) -> None:
        with self._tx() as c:
            cur = c.execute(
                "UPDATE designs SET status=?, state_json=?, updated_at=? WHERE design_id=?",
                (state.status, state.model_dump_json(), state.updated_at.isoformat(), state.design_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"design {state.design_id} not in database")

    def get(self, design_id: str) -> DesignState:
        row = self._conn.execute("SELECT state_json FROM designs WHERE design_id=?", (design_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown design {design_id}")
        return DesignState.model_validate_json(row["state_json"])

    def find(self, id_or_slug: str) -> DesignState:
        row = self._conn.execute(
            "SELECT state_json FROM designs WHERE design_id=? OR slug=?", (id_or_slug, id_or_slug)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown design {id_or_slug!r}")
        return DesignState.model_validate_json(row["state_json"])

    def run_id_of(self, design_id: str) -> str:
        row = self._conn.execute("SELECT run_id FROM designs WHERE design_id=?", (design_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown design {design_id}")
        return str(row["run_id"])

    def by_status(self, statuses: Status | Iterable[Status], run_id: str | None = None) -> list[DesignState]:
        sts = [statuses] if isinstance(statuses, str) else list(statuses)
        q = f"SELECT state_json FROM designs WHERE status IN ({','.join('?' * len(sts))})"
        params: list[Any] = list(sts)
        if run_id:
            q += " AND run_id=?"
            params.append(run_id)
        q += " ORDER BY created_at, slug"
        return [DesignState.model_validate_json(r["state_json"]) for r in self._conn.execute(q, params)]

    def all(self, run_id: str | None = None) -> list[DesignState]:
        if run_id:
            rows = self._conn.execute(
                "SELECT state_json FROM designs WHERE run_id=? ORDER BY created_at, slug", (run_id,))
        else:
            rows = self._conn.execute("SELECT state_json FROM designs ORDER BY created_at, slug")
        return [DesignState.model_validate_json(r["state_json"]) for r in rows]

    def slugs(self) -> set[str]:
        return {str(r[0]) for r in self._conn.execute("SELECT slug FROM designs")}

    def counts_by_status(self, run_id: str | None = None) -> dict[str, int]:
        if run_id:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM designs WHERE run_id=? GROUP BY status", (run_id,))
        else:
            rows = self._conn.execute("SELECT status, COUNT(*) FROM designs GROUP BY status")
        return {str(r[0]): int(r[1]) for r in rows}

    # ---------------------------------------------------------------- events
    def log_event(self, design_id: str | None, stage: str, action: str,
                  detail: str | dict[str, Any] | None = None, traceback: str | None = None) -> None:
        if isinstance(detail, dict):
            detail = json.dumps(detail, default=str)
        with self._tx() as c:
            c.execute(
                "INSERT INTO events(design_id, ts, stage, action, detail, traceback) VALUES (?,?,?,?,?,?)",
                (design_id, utcnow().isoformat(), stage, action, detail, traceback),
            )

    def events(self, design_id: str | None = None, action: str | None = None,
               limit: int = 1000) -> list[EventRow]:
        q = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if design_id:
            q += " AND design_id=?"
            params.append(design_id)
        if action:
            q += " AND action=?"
            params.append(action)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [EventRow(**dict(r)) for r in self._conn.execute(q, params)]

    # ------------------------------------------------------------ embeddings
    def put_embedding(self, design_id: str, path: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).ravel()
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO embeddings(design_id, path, dim, vector) VALUES (?,?,?,?)",
                (design_id, path, int(vec.size), vec.tobytes()),
            )

    def delete_embedding(self, design_id: str) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM embeddings WHERE design_id=?", (design_id,))

    def embeddings(self) -> list[tuple[str, str, np.ndarray]]:
        out = []
        for r in self._conn.execute("SELECT design_id, path, dim, vector FROM embeddings"):
            out.append((str(r["design_id"]), str(r["path"]),
                        np.frombuffer(r["vector"], dtype=np.float32, count=int(r["dim"]))))
        return out


def _run_row(row: sqlite3.Row) -> RunRow:
    return RunRow(
        run_id=row["run_id"], created_at=row["created_at"], niche=row["niche"],
        auto_niche=bool(row["auto_niche"]), n=int(row["n"]), dry_run=bool(row["dry_run"]),
        prompt_done=bool(row["prompt_done"]),
    )
