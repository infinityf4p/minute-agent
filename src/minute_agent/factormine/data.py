"""Universe, labels, style exposures and lagged daily variables for factor mining.

Decision at bar d (14:50). The label is the return from the VWAP of bars d+1..240 on day t to the same window on
the next trading day (adjusted), minus the cross-sectional mean of eligible stocks. A second label is the overnight
return (close t -> open t+1). Daily variables are those known at the decision time: rows of earlier days, plus
today's pre_close and the return from pre_close to the close of bar d.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta

import numpy as np
import polars as pl

from minute_agent import split
from minute_agent.data import store
from minute_agent.factormine.config import cache_dir

DAILY_VARS = ["ret_td", "ret_1d", "ret_5d", "ret_20d", "vol_20d", "turn_1d", "turn_20d", "logmv", "amt_20d"]
STYLES = ["logmv", "ret_td", "ret_1d", "ret_5d", "ret_20d", "vol_20d", "turn_20d"]
DAILY_VAR_HELP = {
    "ret_td": "return from pre_close to the close of the decision bar today",
    "ret_1d": "yesterday's daily return", "ret_5d": "return over the 5 days before today",
    "ret_20d": "return over the 20 days before today", "vol_20d": "std of daily returns, 20 days before today",
    "turn_1d": "yesterday's turnover (% of float)", "turn_20d": "mean turnover, 20 days before today",
    "logmv": "log float market value (yesterday)", "amt_20d": "log mean daily amount, 20 days before today",
}


def months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(y: int, m: int) -> tuple[date, date]:
    a = date(y, m, 1)
    b = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
    return a, b


def shift_months(d: date, k: int) -> date:
    y, m = d.year, d.month - k
    while m <= 0:
        y, m = y - 1, m + 12
    return date(y, m, 1)


def _clip_unlocked(d: date) -> date:
    """Last readable date: 2023-12-31 while locked; the holdout end inside split.final_test_unlock."""
    return min(d, split.max_readable_date())


def universe(cfg: dict) -> list[str]:
    """Fixed stock list chosen from data before the first mining year (no look-ahead, no survivor filter)."""
    p = cache_dir(cfg) / "universe.json"
    if p.exists():
        return json.loads(p.read_text())
    a, b = cfg["universe_ref"]
    d = store.read_daily(cfg["dataset"], a, b, columns=["symbol", "date", "amount", "is_st", "is_suspended"])
    sec = store.read_security(cfg["dataset"])
    g = (d.filter(~pl.col("is_suspended")).group_by("symbol")
         .agg(pl.col("amount").mean().alias("adv"), pl.len().alias("n"), pl.col("is_st").last().alias("st"))
         .join(sec.select("symbol", "board", "list_date"), on="symbol", how="left")
         .filter((pl.col("n") >= 40) & ~pl.col("st") & (pl.col("board") != "bse")
                 & (pl.col("list_date") <= b - timedelta(days=180)))
         .sort("adv"))
    syms = g["symbol"].to_list()
    if cfg["n_symbols"] and cfg["n_symbols"] < len(syms):
        rng = random.Random(cfg["universe_seed"])
        k = cfg["n_symbols"] // 3
        thirds = [syms[i * len(syms) // 3:(i + 1) * len(syms) // 3] for i in range(3)]
        syms = sorted(s for t in thirds for s in rng.sample(t, k))
    p.write_text(json.dumps(syms))
    return syms


def base_panel(cfg: dict, start: date, end: date) -> pl.DataFrame:
    """One row per (symbol, trading date) in [start, end]: mask, labels, daily variables."""
    p = cache_dir(cfg) / f"base_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    ds, d, syms = cfg["dataset"], cfg["decision_bar"], universe(cfg)
    r0, r1 = start - timedelta(days=70), _clip_unlocked(end + timedelta(days=12))
    daily = store.read_daily(ds, r0, r1, symbols=syms)
    cal = store.read_calendar(ds, r0, r1)
    nxt = pl.DataFrame({"date": cal[:-1], "next_date": cal[1:]})
    sec = store.read_security(ds).select("symbol", "list_date")

    rows = []
    for y, m in months(start, min(end + timedelta(days=12), r1)):
        a, b = month_bounds(y, m)
        mm = store.read_minute(ds, max(a, r0), min(b, r1), symbols=syms,
                               columns=["symbol", "date", "bidx", "close", "volume", "amount"])
        rows.append(mm.group_by("symbol", "date").agg(
            pl.col("close").filter(pl.col("bidx") <= d).last().alias("close_d"),
            (pl.col("amount").filter(pl.col("bidx") > d).sum()
             / pl.col("volume").filter(pl.col("bidx") > d).sum()).alias("vwap_tail")))
    intra = pl.concat(rows)

    x = (daily.join(intra, on=["symbol", "date"], how="left").join(nxt, on="date", how="left")
         .join(sec, on="symbol", how="left").sort("symbol", "date"))
    adj_close = pl.col("close") * pl.col("adj_factor")
    o = pl.col("symbol")
    x = x.with_columns(
        (pl.col("vwap_tail") * pl.col("adj_factor")).alias("_px"),
        adj_close.alias("_ac"),
        (adj_close / adj_close.shift(1).over(o) - 1).alias("_ret"),
    ).with_columns(
        pl.col("date").shift(-1).over(o).alias("_d1"),
        pl.col("_px").shift(-1).over(o).alias("_px1"),
        (pl.col("open") * pl.col("adj_factor")).shift(-1).over(o).alias("_ao1"),
    ).with_columns(
        pl.when(pl.col("_d1") == pl.col("next_date")).then(pl.col("_px1") / pl.col("_px") - 1).alias("label_raw"),
        pl.when(pl.col("_d1") == pl.col("next_date")).then(pl.col("_ao1") / pl.col("_ac") - 1).alias("label_on_raw"),
        (pl.col("close_d") / pl.col("pre_close") - 1).alias("ret_td"),
        pl.col("_ret").shift(1).over(o).alias("ret_1d"),
        (pl.col("_ac").shift(1) / pl.col("_ac").shift(6) - 1).over(o).alias("ret_5d"),
        (pl.col("_ac").shift(1) / pl.col("_ac").shift(21) - 1).over(o).alias("ret_20d"),
        pl.col("_ret").shift(1).rolling_std(20, min_samples=10).over(o).alias("vol_20d"),
        pl.col("turnover").shift(1).over(o).alias("turn_1d"),
        pl.col("turnover").shift(1).rolling_mean(20, min_samples=10).over(o).alias("turn_20d"),
        pl.col("float_mv").shift(1).over(o).log().alias("logmv"),
        pl.col("amount").shift(1).rolling_mean(20, min_samples=10).over(o).log().alias("amt_20d"),
    )
    at_limit = ((pl.col("up_limit").is_not_null() & (pl.col("close_d") >= pl.col("up_limit") - 0.005))
                | (pl.col("down_limit").is_not_null() & (pl.col("close_d") <= pl.col("down_limit") + 0.005)))
    x = x.filter(pl.col("date").is_between(start, end)).with_columns(
        (~pl.col("is_st") & ~pl.col("is_suspended") & pl.col("close_d").is_not_null()
         & pl.col("label_raw").is_not_null() & pl.col("label_raw").is_finite()
         & ((pl.col("date") - pl.col("list_date")).dt.total_days() >= cfg["min_listed_days"])
         & ~at_limit.fill_null(False)).fill_null(False).alias("mask"))
    for lab in ("label", "label_on"):
        raw = pl.col(f"{lab}_raw")
        x = x.with_columns(pl.when(pl.col("mask")).then(raw - raw.filter(pl.col("mask")).mean().over("date"))
                           .alias(lab))
    out = x.select(["symbol", "date", "mask", "label", "label_on"] + DAILY_VARS).with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in DAILY_VARS])
    out.write_parquet(p)
    return out


def period_dates(cfg: dict, periods: list[tuple[date, date]]) -> list[date]:
    out = []
    for a, b in periods:
        out += store.read_calendar(cfg["dataset"], a, b)
    return out


def fold_of(dates: np.ndarray, periods: list[tuple[date, date]]) -> np.ndarray:
    f = np.full(len(dates), -1)
    for i, (a, b) in enumerate(periods):
        f[(dates >= np.datetime64(a)) & (dates <= np.datetime64(b))] = i
    return f
