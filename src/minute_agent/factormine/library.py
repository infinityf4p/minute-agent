"""Factor library of one session and the append-only ledger of every candidate.

Admission (evaluator only, thresholds in config.THRESHOLDS):
  quality   coverage >= min_coverage, |res_t| >= min_res_t, every mining fold has the sign of the mean
  distinct  |corr| < max_corr with every member -> admitted
            exactly one conflicting member and |res_ic| >= replace_ratio x its |res_ic| -> replaces it
            otherwise rejected as redundant
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import polars as pl

from minute_agent.factormine.config import THRESHOLDS

KEYS = ["symbol", "date"]


class Library:
    def __init__(self, d: Path):
        self.dir = d
        d.mkdir(parents=True, exist_ok=True)
        self.meta_path, self.values_path = d / "library.json", d / "library_values.parquet"
        self.members: list[dict] = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else []
        self.values = (pl.read_parquet(self.values_path) if self.values_path.exists()
                       else pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date}))

    def save(self) -> None:
        self.meta_path.write_text(json.dumps(self.members, indent=1, default=str))
        self.values.write_parquet(self.values_path)

    def ids(self) -> list[str]:
        return [m["id"] for m in self.members]

    def decide(self, m: dict, corrs: dict[str, float]) -> tuple[str, str | None, str]:
        """Return (status, replaced member id, reason)."""
        if "error" in m:
            return "rejected_invalid", None, m["error"]
        if m["coverage"] < THRESHOLDS["min_coverage"]:
            return "rejected_weak", None, f"coverage {m['coverage']:.2f} < {THRESHOLDS['min_coverage']}"
        t = m.get("res_t")
        if t is None or not np.isfinite(t) or abs(t) < THRESHOLDS["min_res_t"]:
            return "rejected_weak", None, f"|residual IC t| {abs(t or 0):.2f} < {THRESHOLDS['min_res_t']}"
        if THRESHOLDS["fold_sign"]:
            s = np.sign(m["res_ic"])
            if any(not np.isfinite(x) or np.sign(x) != s for x in m["res_ic_folds"]):
                return "rejected_weak", None, f"residual IC changes sign across folds {np.round(m['res_ic_folds'], 4).tolist()}"
        conflicts = {k: v for k, v in corrs.items() if abs(v) >= THRESHOLDS["max_corr"]}
        if not conflicts:
            return "admitted", None, "ok"
        if len(conflicts) == 1:
            (mid, c), = conflicts.items()
            old = next(x for x in self.members if x["id"] == mid)
            if not old.get("fixed") and abs(m["res_ic"]) >= THRESHOLDS["replace_ratio"] * abs(old["metrics"]["res_ic"]):
                return "replaced", mid, f"replaces {mid} (corr {c:+.2f})"
        return "rejected_redundant", None, "corr with " + ", ".join(f"{k} {v:+.2f}" for k, v in conflicts.items())

    def found(self) -> list[dict]:
        """Members found by the search (not the fixed reference seeds)."""
        return [x for x in self.members if not x.get("fixed")]

    def add(self, mid: str, spec: dict, m: dict, o: pl.DataFrame, n: int, replaced: str | None,
            fixed: bool = False) -> None:
        if replaced:
            self.members = [x for x in self.members if x["id"] != replaced]
            self.values = self.values.drop(replaced)
        self.members.append({"id": mid, "spec": spec, "metrics": m, "admitted_at": n, "fixed": fixed})
        col = o.select(KEYS + [pl.col("o").alias(mid)])
        self.values = col if self.values.width == 2 and self.values.height == 0 else \
            self.values.join(col, on=KEYS, how="full", coalesce=True)
        self.save()


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(path, timeout=60)
        self.c.row_factory = sqlite3.Row
        self.c.execute("""CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, arm TEXT, seed INTEGER, n INTEGER,
            created_at TEXT, mechanism TEXT, name TEXT, spec_key TEXT, spec_json TEXT, status TEXT,
            reason TEXT, metrics_json TEXT, secs REAL)""")
        self.c.commit()

    def add(self, **kw) -> None:
        kw.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        cols = ",".join(kw)
        self.c.execute(f"INSERT INTO candidates ({cols}) VALUES ({','.join('?' * len(kw))})", list(kw.values()))
        self.c.commit()

    def rows(self, session: str) -> list[sqlite3.Row]:
        return self.c.execute("SELECT * FROM candidates WHERE session=? ORDER BY id", (session,)).fetchall()
