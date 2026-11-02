"""Execution price tables, precomputed from minute bars and cached per month.

A decision taken after bar d closes is executed over the bars after it:
    next_vwap  bar d+1, volume-weighted price (amount / volume)
    next_open  bar d+1 open
    twap5      bars d+1..d+5, mean of the bars' VWAPs
    twap10     bars d+1..d+10
Never in the signal bar itself. Bars with zero volume don't fill.

can_buy  = some execution bar traded below the up limit (not locked limit-up)
can_sell = some execution bar traded above the down limit
avail    = shares traded in the execution bars (the participation cap applies to it)
sigma    = std of 1-minute log returns over the 30 bars up to d (for the impact model)

The auction table gives, per stock-day, the opening auction (bar 0) and closing
auction (bar 240) prices and volumes, with the same buy/sell flags.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from minute_agent import split
from minute_agent.data import store
from minute_agent.paths import cache_root

STYLES = {"next_vwap": 1, "next_open": 1, "twap5": 5, "twap10": 10}
EPS = 1e-6


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        m0 = date(y, m, 1)
        m1 = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
        yield y, m, max(m0, start), min(m1, end)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _limits(dataset: str, m0: date, m1: date, symbols) -> pl.DataFrame:
    return store.read_daily(dataset, m0, m1, symbols=symbols,
                            columns=["symbol", "date", "up_limit", "down_limit"])


def _month_exec(dataset: str, m0: date, m1: date, decisions: list[int], style: str,
                symbols: list[str] | None) -> pl.DataFrame:
    k = STYLES[style]
    need = sorted({b for d in decisions for b in range(max(0, d - 29), min(240, d + k) + 1)})
    m = store.read_minute(dataset, m0, m1, symbols=symbols,
                          columns=["symbol", "date", "bidx", "open", "high", "low", "close", "volume", "amount"])
    m = m.filter(pl.col("bidx").is_in(need)).sort("symbol", "date", "bidx").with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().over("symbol", "date").alias("lr"),
        pl.when(pl.col("volume") > 0).then(pl.col("amount") / pl.col("volume")).alias("bvwap"),
    ).join(_limits(dataset, m0, m1, symbols), on=["symbol", "date"], how="left")
    traded = pl.col("volume") > 0
    up = pl.col("up_limit").fill_null(float("inf"))
    dn = pl.col("down_limit").fill_null(0.0)
    if style == "next_open":
        px = pl.col("open").filter(traded).first()
    elif style == "next_vwap":
        px = pl.col("amount").filter(traded).sum() / pl.col("volume").filter(traded).sum()
    else:
        px = pl.col("bvwap").filter(traded).mean()
    out = []
    for d in decisions:   # one decision bar at a time keeps the peak memory to one month of bars
        ex = m.filter(pl.col("bidx").is_between(d + 1, min(240, d + k))).group_by("symbol", "date").agg(
            px.alias("px"),
            pl.col("volume").sum().alias("avail"),
            (traded & (pl.col("low") < up - EPS)).any().alias("can_buy"),
            (traded & (pl.col("high") > dn + EPS)).any().alias("can_sell"),
        )
        sg = m.filter(pl.col("bidx").is_between(max(1, d - 29), d)).group_by("symbol", "date").agg(
            pl.col("lr").std().alias("sigma"))
        out.append(ex.join(sg, on=["symbol", "date"], how="left").with_columns(pl.lit(d, dtype=pl.Int16).alias("d")))
    del m
    return pl.concat(out).select("symbol", "date", "d", "px", "avail", "can_buy", "can_sell", "sigma").with_columns(
        pl.col("px").fill_nan(None)).sort("symbol", "date", "d")


def exec_table(dataset: str, start: date, end: date, decisions: list[int], style: str,
               symbols: list[str] | None = None, universe_tag: str = "all") -> pl.DataFrame:
    """Execution table for the given decision bars, cached per month."""
    if style not in STYLES:
        raise ValueError(f"unknown execution style {style!r}; choose from {sorted(STYLES)}")
    split.check_range(start, end)
    decisions = sorted(set(int(d) for d in decisions))
    root = cache_root() / "exec" / dataset / universe_tag / f"{style}-{_key(decisions)}"
    frames = []
    for y, m, m0, m1 in _months(start, end):
        f = root / f"{y}-{m:02d}.parquet"
        full0 = date(y, m, 1)
        full1 = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
        if not f.exists():
            if not split.partition_allowed(y, m):
                continue
            t = _month_exec(dataset, full0, min(full1, split.max_readable_date()), decisions, style, symbols)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            t.write_parquet(tmp)
            tmp.replace(f)
        frames.append(pl.read_parquet(f).filter(pl.col("date").is_between(m0, m1)))
    return pl.concat(frames) if frames else pl.DataFrame()


def auction_table(dataset: str, start: date, end: date, symbols: list[str] | None = None,
                  universe_tag: str = "all") -> pl.DataFrame:
    """Opening (bar 0) and closing (bar 240) auction prices, volumes and tradability per stock-day."""
    split.check_range(start, end)
    root = cache_root() / "exec" / dataset / universe_tag / "auction"
    frames = []
    for y, m, m0, m1 in _months(start, end):
        f = root / f"{y}-{m:02d}.parquet"
        if not f.exists():
            if not split.partition_allowed(y, m):
                continue
            full0 = date(y, m, 1)
            full1 = min((date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1),
                        split.max_readable_date())
            mm = store.read_minute(dataset, full0, full1, symbols=symbols,
                                   columns=["symbol", "date", "bidx", "high", "low", "close", "volume"])
            mm = mm.filter(pl.col("bidx").is_in([0, 240])).join(_limits(dataset, full0, full1, symbols),
                                                                on=["symbol", "date"], how="left")
            up, dn = pl.col("up_limit").fill_null(float("inf")), pl.col("down_limit").fill_null(0.0)
            t = mm.group_by("symbol", "date").agg(
                pl.col("close").filter(pl.col("bidx") == 0).first().alias("open_px"),
                pl.col("volume").filter(pl.col("bidx") == 0).first().alias("open_vol"),
                pl.col("close").filter(pl.col("bidx") == 240).first().alias("close_px"),
                pl.col("volume").filter(pl.col("bidx") == 240).first().alias("close_vol"),
                ((pl.col("close") < up - EPS) & (pl.col("volume") > 0)).filter(pl.col("bidx") == 0).first().alias("open_can_buy"),
                ((pl.col("close") > dn + EPS) & (pl.col("volume") > 0)).filter(pl.col("bidx") == 0).first().alias("open_can_sell"),
                ((pl.col("close") < up - EPS) & (pl.col("volume") > 0)).filter(pl.col("bidx") == 240).first().alias("close_can_buy"),
                ((pl.col("close") > dn + EPS) & (pl.col("volume") > 0)).filter(pl.col("bidx") == 240).first().alias("close_can_sell"),
            ).sort("symbol", "date")
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            t.write_parquet(tmp)
            tmp.replace(f)
        frames.append(pl.read_parquet(f).filter(pl.col("date").is_between(m0, m1)))
    return pl.concat(frames) if frames else pl.DataFrame()
