"""Intraday stage: compute DSL nodes and code units month by month, cached per unit and month.

    cache/factormine/<cfg>/intraday/<unit key>/<YYYY-MM>.parquet   (columns symbol, date, values)

A DSL node gives one column "v". A code unit (agent-written `def feature(m, daily, d)`) gives 1..3 named columns;
it runs in the sandbox of minute_agent.features.extra_ops and must pass the look-ahead check first.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import polars as pl

from minute_agent.data import store
from minute_agent.factormine import data, dsl
from minute_agent.factormine.config import cache_dir
from minute_agent.features import leakage, ops
from minute_agent.features.extra_ops import CustomFeatureError, compile_feature

KEYS = ["symbol", "date"]
MINUTE_COLS = ["symbol", "date", "ts", "bidx", "open", "high", "low", "close", "volume", "amount"]
CODE_MAX_SECONDS_PER_MONTH = 60.0
MAX_CODE_COLS = 3


class UnitError(RuntimeError):
    pass


def node_unit(node: dict) -> dict:
    return {"kind": "dsl", "node": node, "key": "n" + dsl.h(node)}


def code_unit(code: str) -> dict:
    return {"kind": "code", "code": code, "key": "c" + dsl.h(code)}


def run_code(m: pl.DataFrame, daily: pl.DataFrame, d: int, code: str = "") -> pl.DataFrame:
    """Run agent code; return symbol, date and its 1..3 numeric columns for every stock-day in m."""
    fn = compile_feature(code)
    out = fn(m, daily, d)
    if not isinstance(out, pl.DataFrame) or any(k not in out.columns for k in KEYS):
        raise CustomFeatureError("feature() must return a polars DataFrame with symbol and date columns")
    vals = [c for c in out.columns if c not in KEYS]
    if not vals or len(vals) > MAX_CODE_COLS:
        raise CustomFeatureError(f"feature() must return 1..{MAX_CODE_COLS} value columns")
    bad = [c for c in vals if not c.isidentifier() or c.startswith("_") or c in data.DAILY_VARS]
    if bad:
        raise CustomFeatureError(f"bad output column names {bad} (identifiers, not daily variable names)")
    out = out.select(KEYS + [pl.col(c).cast(pl.Float64, strict=False) for c in vals])
    if out.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise CustomFeatureError("feature() returned duplicate (symbol, date) rows")
    return m.select(KEYS).unique().join(out, on=KEYS, how="left")


def code_columns(cfg: dict, code: str) -> list[str]:
    """Output columns of a code unit, from a small sample (also validates that it runs)."""
    m, daily = _sample(cfg, n_sym=40)
    out = run_code(ops.prepare_minute(ops.cutoff(m, cfg["decision_bar"])), daily, cfg["decision_bar"], code=code)
    return [c for c in out.columns if c not in KEYS]


def _sample(cfg: dict, n_sym: int = 150, a: date = date(2020, 3, 2), b: date = date(2020, 3, 20)):
    syms = data.universe(cfg)[:: max(1, len(data.universe(cfg)) // n_sym)][:n_sym]
    m = store.read_minute(cfg["dataset"], a, b, symbols=syms, columns=MINUTE_COLS)
    daily = store.read_daily(cfg["dataset"], a - timedelta(days=70), b, symbols=syms)
    return m, daily


def check_code_leakage(cfg: dict, code: str) -> None:
    m, daily = _sample(cfg)
    dates = leakage.pick_check_dates(m["date"].unique().to_list(), k=2)
    leakage.check_op(run_code, {"code": code}, m, daily, cfg["decision_bar"], dates, apply_cutoff=True)


def _path(cfg: dict, key: str, y: int, m: int):
    return cache_dir(cfg) / "intraday" / key / f"{y}-{m:02d}.parquet"


def compute(cfg: dict, units: list[dict], start: date, end: date) -> dict[str, pl.DataFrame | Exception]:
    """Values of every unit for trading days in [start, end]. A failing unit maps to its exception."""
    d, syms = cfg["decision_bar"], data.universe(cfg)
    uniq = {u["key"]: u for u in units}
    failed: dict[str, Exception] = {}
    for y, mo in data.months(start, end):
        todo = [u for k, u in uniq.items() if k not in failed and not _path(cfg, k, y, mo).exists()]
        if not todo:
            continue
        a, b = data.month_bounds(y, mo)
        raw = store.read_minute(cfg["dataset"], a, b, symbols=syms, columns=MINUTE_COLS)
        nodes = [u for u in todo if u["kind"] == "dsl"]
        if nodes:
            bars = dsl.prepare_bars(raw, d)
            try:
                res = dsl.eval_nodes(bars, {u["key"]: u["node"] for u in nodes}, d)
                outs = {u["key"]: res.select(KEYS + [pl.col(u["key"]).alias("v")]) for u in nodes}
            except Exception:       # isolate the failing node
                outs = {}
                for u in nodes:
                    try:
                        outs[u["key"]] = dsl.eval_nodes(bars, {"v": u["node"]}, d)
                    except Exception as e:
                        failed[u["key"]] = UnitError(f"intraday node failed: {e}")
            for k, o in outs.items():
                _write(o, _path(cfg, k, y, mo))
        codes = [u for u in todo if u["kind"] == "code"]
        if codes:
            m_cut = ops.prepare_minute(ops.cutoff(raw, d))
            daily = store.read_daily(cfg["dataset"], a - timedelta(days=70), b, symbols=syms)
            for u in codes:
                t0 = time.time()
                try:
                    o = run_code(m_cut, daily, d, code=u["code"])
                except Exception as e:
                    failed[u["key"]] = UnitError(f"code failed in {y}-{mo:02d}: {type(e).__name__}: {e}"[:600])
                    continue
                if time.time() - t0 > CODE_MAX_SECONDS_PER_MONTH:
                    failed[u["key"]] = UnitError(f"code too slow: {time.time() - t0:.0f}s for one month "
                                                 f"(limit {CODE_MAX_SECONDS_PER_MONTH:.0f}s)")
                    continue
                _write(o, _path(cfg, u["key"], y, mo))
    out: dict[str, pl.DataFrame | Exception] = dict(failed)
    for k in uniq:
        if k in failed:
            continue
        parts = [pl.read_parquet(_path(cfg, k, y, mo)) for y, mo in data.months(start, end)]
        out[k] = pl.concat(parts, how="vertical_relaxed").filter(pl.col("date").is_between(start, end))
    return out


def _write(df: pl.DataFrame, p) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, p)
