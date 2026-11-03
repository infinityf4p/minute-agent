"""Atomic feature cache: one file per (operator, single parameter value, decision time, month).

A spec such as {op: ret, window_min: [5, 30]} is the union of the atoms
{op: ret, window_min: [5]} and {op: ret, window_min: [30]}; every operator with a
list parameter names its output columns per list element, so the union of the
atoms' columns is exactly the spec's output. Caching atoms means any subset the
agent picks is already cached once each atom was computed.

All missing atoms of one operator are computed in a single pass (the operator run
with the union of their parameters) and then split into atom files.

    cache/atoms/<dataset>/dec<HHMM>[/u-<tag>]/<atom_key>/<YYYY-MM>.parquet
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from minute_agent import split
from minute_agent.data import store
from minute_agent.features import leakage, ops
from minute_agent.features.cache import DAILY_BUFFER_DAYS, _month_end, _months, normalize_spec
from minute_agent.market import load_market
from minute_agent.paths import cache_root

KEYS = ops.KEYS

# list parameters that are split into atoms, and the output columns of one atom
LIST_PARAMS = {
    "ret": ["window_min"], "realized_vol": ["window_min"], "volume_profile": ["buckets"],
    "return_moments": ["window_min"], "pv_corr": ["window_min"], "signed_volume": ["window_min"],
    "daily_stats": ["windows"], "daily_lag": ["fields", "lags"],
}


def _cols(name: str, p: dict) -> list[str] | None:
    """Columns produced by one atom (None = all columns of the operator output)."""
    if name == "ret":
        return [f"ret_{p['window_min'][0]}"]
    if name == "realized_vol":
        return [f"rv_{p['window_min'][0]}"]
    if name == "volume_profile":
        return [f"vp_{p['buckets'][0]}"]
    if name == "return_moments":
        w = p["window_min"][0]
        return [f"skew_{w}", f"kurt_{w}"]
    if name == "pv_corr":
        w = p["window_min"][0]
        return [f"pvabs_{w}", f"pvsgn_{w}"]
    if name == "signed_volume":
        return [f"sv_{p['window_min'][0]}"]
    if name == "daily_stats":
        w = int(p["windows"][0])
        return [f"ds_vol_{w}", f"ds_illiq_{w}", f"ds_offhigh_{w}", f"ds_offlow_{w}"]
    if name == "daily_lag":
        return [f"dl_{p['fields'][0]}_{p['lags'][0]}"]
    return None


def atomize(spec: dict) -> list[dict]:
    name, params = normalize_spec(spec)
    lp = LIST_PARAMS.get(name)
    if not lp:
        return [{"op": name, **params}]
    fixed = {k: v for k, v in params.items() if k not in lp}
    combos = itertools.product(*[params[k] for k in lp])
    return [{"op": name, **fixed, **{k: [v] for k, v in zip(lp, c)}} for c in combos]


def atom_key(atom: dict, decision_time: str) -> str:
    name, params = normalize_spec(atom)
    _, version = ops.REGISTRY[name]
    blob = json.dumps({"op": name, "params": params, "v": version, "dec": decision_time}, sort_keys=True)
    return f"{name}-{hashlib.sha1(blob.encode()).hexdigest()[:10]}"


def _union(atoms: list[dict]) -> dict:
    name, params = normalize_spec(atoms[0])
    lp = LIST_PARAMS[name]
    out = {k: v for k, v in params.items() if k not in lp}
    for k in lp:
        vals = []
        for a in atoms:
            for v in a[k]:
                if v not in vals:
                    vals.append(v)
        out[k] = vals
    return {"op": name, **out}


def _build_month(dataset, d, y, m, m1, groups, symbols):
    """groups: list of (union_spec or atom spec, [(atom, path)]). One read of the month, one pass per operator."""
    t0 = time.time()
    m0 = date(y, m, 1)
    minute = store.read_minute(dataset, m0, m1, max_bidx=d, symbols=symbols)
    daily = store.read_daily(dataset, m0 - timedelta(days=DAILY_BUFFER_DAYS), m1, symbols=symbols)
    mp = ops.prepare_minute(ops.cutoff(minute, d))
    del minute
    n = 0
    for spec, members in groups:
        name, params = normalize_spec(spec)
        res = ops.compute(name, mp, daily, d, params).filter(pl.col("date").is_between(m0, m1))
        for atom, path in members:
            an, ap = normalize_spec(atom)
            cols = _cols(an, ap)
            part = res.select(KEYS + cols) if cols else res
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            part.write_parquet(tmp, compression="zstd")
            os.replace(tmp, p)
            n += 1
    return {"month": f"{y}-{m:02d}", "atoms": n, "seconds": round(time.time() - t0, 1)}


class AtomCache:
    def __init__(self, dataset: str, market: str = "cn", decision_time: str | None = None,
                 symbols: list[str] | None = None, universe_tag: str | None = None):
        self.dataset = dataset
        self.rules = load_market(market)
        self.decision_time = decision_time or self.rules.decision_time
        self.d = self.rules.decision_bidx(self.decision_time)
        self.symbols = sorted(symbols) if symbols else None
        if self.symbols and not universe_tag:
            raise ValueError("a symbol subset needs a universe_tag")
        self.root = cache_root() / "atoms" / dataset / f"dec{self.decision_time.replace(':', '')}"
        if self.symbols:
            self.root = self.root / f"u-{universe_tag}"

    def _path(self, atom: dict, y: int, m: int) -> Path:
        return self.root / atom_key(atom, self.decision_time) / f"{y}-{m:02d}.parquet"

    def _atoms(self, specs: list[dict]) -> list[dict]:
        seen, out = set(), []
        for s in [{"op": "base"}] + [s for s in specs if s["op"] != "base"]:
            for a in atomize(s):
                k = json.dumps(a, sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    out.append(a)
        return out

    def ensure_checked(self, atom: dict) -> None:
        marker = self.root / atom_key(atom, self.decision_time) / "_leak_check.json"
        if marker.exists():
            return
        name, params = normalize_spec(atom)
        fn, _ = ops.REGISTRY[name]
        months = store.minute_months(self.dataset, allow_locked=split._FINAL_EVAL)
        y, m = months[len(months) // 2]
        start, end = date(y, m, 1), _month_end(y, m)
        minute = store.read_minute(self.dataset, start, end, symbols=self.symbols)
        syms = sorted(minute["symbol"].unique().to_list())[:40]
        minute = minute.filter(pl.col("symbol").is_in(syms))
        daily = store.read_daily(self.dataset, start - timedelta(days=DAILY_BUFFER_DAYS), end, symbols=syms)
        dates = leakage.pick_check_dates(minute["date"].unique().to_list(), k=3)
        leakage.check_op(fn, params, minute, daily, self.d, dates, apply_cutoff=True)
        if name == "custom":
            leakage.check_op(fn, params, minute, daily, self.d, dates, apply_cutoff=False)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"atom": atom, "checked_dates": [str(x) for x in dates]}))

    def build(self, specs: list[dict], start: date, end: date, workers: int = 1, verbose: bool = False) -> dict:
        split.check_range(start, end)
        atoms = self._atoms(specs)
        for a in atoms:
            self.ensure_checked(a)
        on_disk = set(store.minute_months(self.dataset, allow_locked=split._FINAL_EVAL))
        tasks = []
        for y, m in _months(start, end):
            if (y, m) not in on_disk:
                continue
            missing = [a for a in atoms if not self._path(a, y, m).exists()]
            if not missing:
                continue
            by_op: dict[str, list] = {}
            for a in missing:
                by_op.setdefault(a["op"], []).append(a)
            groups = []
            for name, aa in by_op.items():
                if name in LIST_PARAMS:
                    groups.append((_union(aa), [(a, str(self._path(a, y, m))) for a in aa]))
                else:
                    groups += [(a, [(a, str(self._path(a, y, m)))]) for a in aa]
            tasks.append((self.dataset, self.d, y, m, min(_month_end(y, m), end), groups, self.symbols))
        t0 = time.time()
        if split._FINAL_EVAL or workers <= 1 or len(tasks) <= 1:
            results = [_build_month(*t) for t in tasks]
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from minute_agent._mp import init_worker

            threads = max(1, (os.cpu_count() or 4) // workers)
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=init_worker, initargs=(threads,)) as ex:
                results = list(ex.map(_build_month, *zip(*tasks)))
        if verbose:
            for r in results:
                print("  atoms", r, flush=True)
        return {"months_built": len(tasks), "seconds": round(time.time() - t0, 1)}

    def load(self, specs: list[dict], start: date, end: date) -> pl.DataFrame:
        split.check_range(start, end)
        atoms = self._atoms(specs)
        months = []
        for y, m in _months(start, end):
            files = [self._path(a, y, m) for a in atoms]
            if not files[0].exists():
                continue
            if not all(f.exists() for f in files):
                missing = [a for a, f in zip(atoms, files) if not f.exists()]
                raise FileNotFoundError(f"atoms not cached for {y}-{m:02d}: {missing[:3]}; run build first")
            frames = [pl.read_parquet(f) for f in files]
            base = frames[0]
            parts = [base]
            for fr in frames[1:]:
                vals = fr.drop(KEYS)
                same = fr.height == base.height and fr.select(KEYS).equals(base.select(KEYS))
                parts.append(vals if same else base.select(KEYS).join(fr, on=KEYS, how="left").drop(KEYS))
            months.append(pl.concat(parts, how="horizontal").filter(pl.col("date").is_between(start, end)))
        if not months:
            raise FileNotFoundError(f"no cached atoms in {start}..{end}; run build first")
        out = pl.concat(months, how="vertical_relaxed")
        dup = [c for c in out.columns if out.columns.count(c) > 1]
        if dup:
            raise ValueError(f"duplicate feature columns {dup}")
        return out.sort(KEYS)
