"""Append-only SQLite experiment ledger. Every attempt is recorded, including
failures, so the final report can correct for the true number of trials."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from minute_agent.paths import runs_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,              -- running / done / failed
    source TEXT,                       -- human / agent / random
    dataset TEXT, market TEXT, target TEXT, eval_split TEXT,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    ic REAL, rank_ic REAL, icir REAL,
    excess_net_bp REAL, excess_net_ann REAL, excess_net_sharpe REAL, top_turnover REAL,
    metrics_json TEXT,
    t_features REAL, t_panel REAL, t_train REAL, t_eval REAL, t_total REAL,
    evaluator_hash TEXT, git_commit TEXT, git_dirty INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS runs_config_hash ON runs(config_hash);
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_freeze_finished BEFORE UPDATE ON runs
WHEN OLD.status != 'running'
BEGIN SELECT RAISE(ABORT, 'finished runs cannot be changed'); END;
"""

HEADLINE = ["ic", "rank_ic", "icir", "excess_net_bp", "excess_net_ann", "excess_net_sharpe", "top_turnover"]


def default_path() -> Path:
    return runs_root() / "ledger.sqlite"


class Ledger:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def start(self, run_id: str, config: dict, config_hash: str, meta: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id, created_at, status, source, dataset, market, target, eval_split,"
                " config_json, config_hash, evaluator_hash, git_commit, git_dirty)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, datetime.now().isoformat(timespec="seconds"), "running", meta.get("source"),
                 config.get("dataset"), config.get("market"), config.get("target"), config.get("eval_split"),
                 json.dumps(config, sort_keys=True), config_hash, meta.get("evaluator_hash"),
                 meta.get("git_commit"), int(bool(meta.get("git_dirty")))),
            )

    def finish(self, run_id: str, metrics: dict, timings: dict) -> None:
        flat = {k: v for k, v in metrics.items() if k != "series"}
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, status='done', metrics_json=?,"
                + ",".join(f"{k}=?" for k in HEADLINE) + ","
                + "t_features=?, t_panel=?, t_train=?, t_eval=?, t_total=? WHERE run_id=?",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(flat, default=str),
                 *[flat.get(k) for k in HEADLINE],
                 *[timings.get(k) for k in ("features", "panel", "train", "eval", "total")], run_id),
            )

    def fail(self, run_id: str, error: str, timings: dict | None = None) -> None:
        timings = timings or {}
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, status='failed', error=?, t_total=? WHERE run_id=?",
                (datetime.now().isoformat(timespec="seconds"), error[-4000:], timings.get("total"), run_id),
            )

    def rows(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            cur = c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
