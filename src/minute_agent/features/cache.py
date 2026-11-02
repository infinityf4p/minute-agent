"""Parquet feature cache, one file per (operator spec, month).

    cache/features/<dataset>/dec<HHMM>/<op_key>/<YYYY-MM>.parquet

op_key = "<op>-<hash of op name, params, op version, decision time>", so a
changed parameter or operator version never reuses stale values.

Before an operator spec is cached for the first time it must pass the
look-ahead check (features.leakage) on a sample of the data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from minute_agent import split
from minute_agent.data import store
from minute_agent.features import leakage, ops
from minute_agent.market import load_market
from minute_agent.paths import cache_root

DAILY_BUFFER_DAYS = 120   # calendar days of daily history loaded before each month


def normalize_spec(spec: dict) -> tuple[str, dict]:
    spec = dict(spec)
    name = spec.pop("op")
    if name not in ops.REGISTRY:
        raise ValueError(f"unknown feature op {name!r}; known: {sorted(ops.REGISTRY)}")
    params = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in spec.items()}
    return name, params


def op_key(spec: dict, decision_time: str) -> str:
    name, params = normalize_spec(spec)
    _, version = ops.REGISTRY[name]
    blob = json.dumps({"op": name, "params": params, "v": version, "dec": decision_time}, sort_keys=True)
    return f"{name}-{hashlib.sha1(blob.encode()).hexdigest()[:10]}"


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _month_end(y: int, m: int) -> date:
    return (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)


@dataclass
class BuildStats:
    months_built: int = 0
    months_cached: int = 0
    minute_rows: int = 0
    seconds_read: float = 0.0
    seconds_compute: float = 0.0
    seconds_leak_check: float = 0.0
    per_op_seconds: dict = field(default_factory=dict)


class FeatureCache:
    def __init__(self, dataset: str, market: str = "cn", decision_time: str | None = None):
        self.dataset = dataset
        self.rules = load_market(market)
        self.decision_time = decision_time or self.rules.decision_time
        self.d = self.rules.decision_bidx(self.decision_time)
        self.root = cache_root() / "features" / dataset / f"dec{self.decision_time.replace(':', '')}"

    def _path(self, key: str, y: int, m: int) -> Path:
        return self.root / key / f"{y}-{m:02d}.parquet"

    # -- leakage gate -------------------------------------------------------

    def ensure_checked(self, spec: dict, stats: BuildStats) -> None:
        key = op_key(spec, self.decision_time)
        marker = self.root / key / "_leak_check.json"
        if marker.exists():
            return
        t0 = time.time()
        name, params = normalize_spec(spec)
        fn, _ = ops.REGISTRY[name]
        months = store.minute_months(self.dataset)
        y, m = months[len(months) // 2]          # a month in the middle of the readable data
        start, end = date(y, m, 1), _month_end(y, m)
        minute = store.read_minute(self.dataset, start, end)
        syms = sorted(minute["symbol"].unique().to_list())[:40]
        minute = minute.filter(pl.col("symbol").is_in(syms))
        daily = store.read_daily(self.dataset, start - timedelta(days=DAILY_BUFFER_DAYS), end, symbols=syms)
        dates = leakage.pick_check_dates(minute["date"].unique().to_list(), k=3)
        leakage.check_op(fn, params, minute, daily, self.d, dates, apply_cutoff=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"op": name, "params": params, "checked_dates": [str(x) for x in dates]}))
        stats.seconds_leak_check += time.time() - t0

    # -- build / load -------------------------------------------------------

    def build(self, specs: list[dict], start: date, end: date, verbose: bool = False) -> BuildStats:
        split.check_range(start, end)
        specs = [{"op": "base"}] + [s for s in specs if s["op"] != "base"]
        stats = BuildStats()
        for s in specs:
            self.ensure_checked(s, stats)
        keys = [op_key(s, self.decision_time) for s in specs]
        on_disk = set(store.minute_months(self.dataset))
        for y, m in _months(start, end):
            if (y, m) not in on_disk:
                continue
            todo = [(s, k) for s, k in zip(specs, keys) if not self._path(k, y, m).exists()]
            if not todo:
                stats.months_cached += 1
                continue
            t0 = time.time()
            m0, m1 = date(y, m, 1), min(_month_end(y, m), end)
            minute = store.read_minute(self.dataset, m0, m1, max_bidx=self.d)
            daily = store.read_daily(self.dataset, m0 - timedelta(days=DAILY_BUFFER_DAYS), m1)
            mp = ops.prepare_minute(ops.cutoff(minute, self.d))
            stats.seconds_read += time.time() - t0
            stats.minute_rows += minute.height
            for s, k in todo:
                t1 = time.time()
                name, params = normalize_spec(s)
                out = ops.compute(name, mp, daily, self.d, params)
                out = out.filter(pl.col("date").is_between(m0, m1))
                p = self._path(k, y, m)
                p.parent.mkdir(parents=True, exist_ok=True)
                out.write_parquet(p, compression="zstd")
                dt = time.time() - t1
                stats.seconds_compute += dt
                stats.per_op_seconds[name] = stats.per_op_seconds.get(name, 0.0) + dt
            stats.months_built += 1
            if verbose:
                print(f"  features {y}-{m:02d}: {minute.height:,} rows, {time.time() - t0:.1f}s", flush=True)
        return stats

    def load(self, specs: list[dict], start: date, end: date, include_base: bool = True) -> pl.DataFrame:
        split.check_range(start, end)
        if include_base:
            specs = [{"op": "base"}] + [s for s in specs if s["op"] != "base"]
        frames = []
        for s in specs:
            k = op_key(s, self.decision_time)
            files = [self._path(k, y, m) for y, m in _months(start, end) if self._path(k, y, m).exists()]
            if not files:
                raise FileNotFoundError(f"no cached features for {s} in {start}..{end}; run build first")
            frames.append(
                pl.scan_parquet(files).filter(pl.col("date").is_between(start, end)).collect()
            )
        out = frames[0]
        for f in frames[1:]:
            dup = [c for c in f.columns if c in out.columns and c not in ops.KEYS]
            if dup:
                raise ValueError(f"duplicate feature columns {dup}; two specs produce the same names")
            out = out.join(f, on=ops.KEYS, how="left")
        return out.sort(ops.KEYS)
