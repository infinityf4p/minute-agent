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
import os
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
    seconds_wall: float = 0.0          # wall time of the month loop (read/compute are summed over workers)
    per_op_seconds: dict = field(default_factory=dict)


def _build_month(dataset: str, d: int, y: int, m: int, m1: date, todo: list[tuple[dict, str]],
                 symbols: list[str] | None = None) -> dict:
    """Compute the given op specs for one month and write their parquet files.
    Top-level so it can run in a worker process."""
    t0 = time.time()
    m0 = date(y, m, 1)
    minute = store.read_minute(dataset, m0, m1, max_bidx=d, symbols=symbols)
    daily = store.read_daily(dataset, m0 - timedelta(days=DAILY_BUFFER_DAYS), m1, symbols=symbols)
    mp_ = ops.prepare_minute(ops.cutoff(minute, d))
    out = {"month": f"{y}-{m:02d}", "rows": minute.height, "read_s": time.time() - t0, "ops": {}}
    for spec, path in todo:
        t1 = time.time()
        name, params = normalize_spec(spec)
        res = ops.compute(name, mp_, daily, d, params).filter(pl.col("date").is_between(m0, m1))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        res.write_parquet(tmp, compression="zstd")
        os.replace(tmp, p)          # never leave a half-written cache file
        out["ops"][name] = out["ops"].get(name, 0.0) + time.time() - t1
    return out


class FeatureCache:
    def __init__(self, dataset: str, market: str = "cn", decision_time: str | None = None,
                 symbols: list[str] | None = None, universe_tag: str | None = None):
        """symbols/universe_tag: restrict the cache to a fixed symbol subset (e.g. the T0 base pool).
        The tag becomes part of the cache path, so subsets never mix with full-market caches."""
        self.symbols = sorted(symbols) if symbols else None
        if self.symbols and not universe_tag:
            raise ValueError("a symbol subset needs a universe_tag")
        self.dataset = dataset
        self.rules = load_market(market)
        self.decision_time = decision_time or self.rules.decision_time
        self.d = self.rules.decision_bidx(self.decision_time)
        self.root = cache_root() / "features" / dataset / f"dec{self.decision_time.replace(':', '')}"
        if self.symbols:
            self.root = self.root / f"u-{universe_tag}"

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
        minute = store.read_minute(self.dataset, start, end, symbols=self.symbols)
        syms = sorted(minute["symbol"].unique().to_list())[:40]
        minute = minute.filter(pl.col("symbol").is_in(syms))
        daily = store.read_daily(self.dataset, start - timedelta(days=DAILY_BUFFER_DAYS), end, symbols=syms)
        dates = leakage.pick_check_dates(minute["date"].unique().to_list(), k=3)
        leakage.check_op(fn, params, minute, daily, self.d, dates, apply_cutoff=True)
        if name == "custom":
            # agent-written code must not look past the decision bar even on raw data (v2 rule)
            leakage.check_op(fn, params, minute, daily, self.d, dates, apply_cutoff=False)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"op": name, "params": params, "checked_dates": [str(x) for x in dates]}))
        stats.seconds_leak_check += time.time() - t0

    # -- build / load -------------------------------------------------------

    def build(self, specs: list[dict], start: date, end: date, verbose: bool = False,
              workers: int = 1) -> BuildStats:
        """Compute missing (op, month) files. workers > 1 runs months in separate
        processes; each process holds one month of minute data in memory."""
        split.check_range(start, end)
        specs = [{"op": "base"}] + [s for s in specs if s["op"] != "base"]
        stats = BuildStats()
        for s in specs:
            self.ensure_checked(s, stats)
        keys = [op_key(s, self.decision_time) for s in specs]
        on_disk = set(store.minute_months(self.dataset, allow_locked=split._FINAL_EVAL))
        tasks = []
        for y, m in _months(start, end):
            if (y, m) not in on_disk:
                continue
            todo = [(s, str(self._path(k, y, m))) for s, k in zip(specs, keys) if not self._path(k, y, m).exists()]
            if todo:
                tasks.append((self.dataset, self.d, y, m, min(_month_end(y, m), end), todo, self.symbols))
            else:
                stats.months_cached += 1
        t_wall = time.time()
        if split._FINAL_EVAL:
            workers = 1   # the unlock flag lives in this process only
        if workers <= 1 or len(tasks) <= 1:
            results = map(lambda t: _build_month(*t), tasks)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from minute_agent._mp import init_worker

            threads = max(1, (os.cpu_count() or 4) // workers)
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=init_worker, initargs=(threads,))
            results = ex.map(_build_month, *zip(*tasks))
        try:
            for r in results:
                stats.months_built += 1
                stats.minute_rows += r["rows"]
                stats.seconds_read += r["read_s"]
                for name, dt in r["ops"].items():
                    stats.seconds_compute += dt
                    stats.per_op_seconds[name] = stats.per_op_seconds.get(name, 0.0) + dt
                if verbose:
                    print(f"  features {r['month']}: {r['rows']:,} rows, {r['read_s'] + sum(r['ops'].values()):.1f}s",
                          flush=True)
        finally:
            if workers > 1 and len(tasks) > 1:
                ex.shutdown()
        stats.seconds_wall = time.time() - t_wall
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
