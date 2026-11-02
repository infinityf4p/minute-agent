"""Point-in-time universe and forward-return labels.

Labels use unadjusted prices times the cumulative back-adjustment factor:
    overnight_t = open_{t+1} * adj_{t+1} / (close_t * adj_t) - 1
    next_cc_t   = close_{t+1} * adj_{t+1} / (close_t * adj_t) - 1
t+1 is the next exchange trading day. If the stock is suspended or missing on
t+1 the label is null (the evaluator treats it as a stuck, flat position).
Labels near the end of the readable period are null because the next day is
not readable; this is the embargo between validation and the locked test set.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

TARGETS = ("overnight", "next_cc")
LABEL_HORIZON_DAYS = {"overnight": 1, "next_cc": 1}


def next_trading_day(calendar: list[date]) -> pl.DataFrame:
    return pl.DataFrame({"date": calendar[:-1], "next_date": calendar[1:]},
                        schema={"date": pl.Date, "next_date": pl.Date})


def labels(daily: pl.DataFrame, calendar: list[date], target: str) -> pl.DataFrame:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    d = daily.select("symbol", "date", "open", "close", "adj_factor", "is_suspended")
    nxt = d.rename({"date": "next_date", "open": "n_open", "close": "n_close",
                    "adj_factor": "n_adj", "is_suspended": "n_susp"})
    j = d.join(next_trading_day(calendar), on="date", how="inner").join(
        nxt, on=["symbol", "next_date"], how="left"
    )
    exit_px = pl.col("n_open") if target == "overnight" else pl.col("n_close")
    lab = (exit_px * pl.col("n_adj")) / (pl.col("close") * pl.col("adj_factor")) - 1
    return j.select(
        "symbol", "date",
        pl.when(pl.col("n_susp").not_() & ~pl.col("is_suspended")).then(lab).otherwise(None).alias("label"),
    )


def universe(
    daily: pl.DataFrame,
    security: pl.DataFrame,
    calendar: list[date],
    base: pl.DataFrame,
    market_universe: dict,
    filt: dict | None = None,
) -> pl.DataFrame:
    """One row per (symbol, date) that traded before the decision time, with
    in_universe and buyable flags. `base` comes from the `base` feature op."""
    filt = filt or {}
    cal = np.array(calendar, dtype="datetime64[D]")

    d = daily.sort("symbol", "date").with_columns(
        pl.col("amount").rolling_mean(20, min_samples=5).shift(1).over("symbol").alias("adv20"),
    ).select("symbol", "date", "is_st", "is_suspended", "up_limit", "adv20")

    sec = security.select("symbol", "list_date")
    first_idx = np.searchsorted(cal, np.array(sec["list_date"].to_list(), dtype="datetime64[D]"))
    sec = sec.with_columns(pl.Series("list_idx", first_idx, dtype=pl.Int64))
    cal_idx = pl.DataFrame({"date": calendar, "cal_idx": np.arange(len(calendar))},
                           schema={"date": pl.Date, "cal_idx": pl.Int64})

    u = (
        base.select("symbol", "date", "price_dec", "amt_dec")
        .join(d, on=["symbol", "date"], how="left")
        .join(sec, on="symbol", how="left")
        .join(cal_idx, on="date", how="left")
        .with_columns((pl.col("cal_idx") - pl.col("list_idx")).alias("listed_days"))
    )

    ok = pl.lit(True)
    if market_universe.get("exclude_st", True):
        ok = ok & ~pl.col("is_st").fill_null(True)
    if market_universe.get("exclude_suspended", True):
        ok = ok & ~pl.col("is_suspended").fill_null(True)
    ok = ok & (pl.col("listed_days") >= market_universe.get("min_listed_days", 60))
    if "min_price" in filt:
        ok = ok & (pl.col("price_dec") >= float(filt["min_price"]))
    u = u.with_columns(ok.fill_null(False).alias("_ok"))
    if "min_adv_pct" in filt:
        pct = float(filt["min_adv_pct"]) / 100.0
        u = u.with_columns(
            (pl.col("adv20").rank().over("date", "_ok") / pl.len().over("date", "_ok")).alias("_adv_pct")
        ).with_columns((pl.col("_ok") & (pl.col("_adv_pct") >= pct).fill_null(False)).alias("_ok"))

    buyable = pl.col("up_limit").is_null() | (pl.col("price_dec") < pl.col("up_limit") - 1e-6)
    return u.select(
        "symbol", "date",
        pl.col("_ok").alias("in_universe"),
        buyable.alias("buyable"),
    )


def cross_section_norm(df: pl.DataFrame, cols: list[str], how: str) -> pl.DataFrame:
    if how in (None, "none"):
        return df
    if how == "rank":
        return df.with_columns([
            ((pl.col(c).rank().over("date") - 1) / (pl.col(c).count().over("date") - 1).clip(lower_bound=1) - 0.5).alias(c)
            for c in cols
        ])
    if how == "zscore":
        return df.with_columns([
            ((pl.col(c) - pl.col(c).mean().over("date")) / pl.col(c).std().over("date")).clip(-5, 5).alias(c)
            for c in cols
        ])
    raise ValueError(f"unknown cross_section_norm {how!r}")
