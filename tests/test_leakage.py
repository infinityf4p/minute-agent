"""Look-ahead tests: deliberately cheating features must be rejected."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from minute_agent.data import store
from minute_agent.features import ops
from minute_agent.features.cache import FeatureCache
from minute_agent.features.leakage import LeakageError, check_op, pick_check_dates

D = 230  # 14:50


@pytest.fixture(scope="module")
def data(sandbox):
    start, end = date(2018, 8, 1), date(2018, 8, 31)
    minute = store.read_minute("tiny", start, end)
    daily = store.read_daily("tiny", start - timedelta(days=120), end)
    dates = pick_check_dates(minute["date"].unique().to_list(), k=3)
    return minute, daily, dates


# --- cheating operators ----------------------------------------------------


def cheat_next_day_open(m, daily, d):
    """Uses tomorrow's first bar: classic cross-day look-ahead inside a vectorized op."""
    first = m.filter(pl.col("bidx") == 0).select("symbol", "date", pl.col("close").alias("o"))
    first = first.sort("symbol", "date").with_columns(pl.col("o").shift(-1).over("symbol").alias("next_open"))
    last = m.group_by("symbol", "date").agg(pl.col("close").last().alias("c"))
    return last.join(first, on=["symbol", "date"]).select(
        "symbol", "date", (pl.col("next_open") / pl.col("c")).log().alias("peek")
    )


def cheat_today_daily_close(m, daily, d):
    """Uses today's daily close, which is only known after 15:00."""
    keys = m.select("symbol", "date").unique()
    return keys.join(daily.select("symbol", "date", pl.col("close").alias("peek_close")),
                     on=["symbol", "date"], how="left")


def cheat_closing_auction(m, daily, d):
    """Reads the 15:00 closing-auction bar directly."""
    return m.group_by("symbol", "date").agg(
        pl.col("close").filter(pl.col("bidx") == 240).last().alias("peek_auction"),
        pl.col("close").last().alias("last_seen"),
    )


def cheat_full_sample_zscore(m, daily, d):
    """Normalizes with mean/std of the whole sample, including the future."""
    r = m.group_by("symbol", "date").agg(pl.col("close").last().alias("c"))
    return r.with_columns(((pl.col("c") - pl.col("c").mean()) / pl.col("c").std()).alias("z")).drop("c")


# --- tests -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(n for n in ops.REGISTRY))
def test_registered_ops_pass(data, name):
    minute, daily, dates = data
    defaults = {"daily_lag": {"fields": ["ret", "turnover"], "lags": [1, 5, 20]}}
    check_op(ops.REGISTRY[name][0], defaults.get(name, {}), minute, daily, D, dates)


@pytest.mark.parametrize("cheat", [cheat_next_day_open, cheat_today_daily_close, cheat_full_sample_zscore])
def test_cheating_ops_are_caught(data, cheat):
    minute, daily, dates = data
    with pytest.raises(LeakageError):
        check_op(cheat, {}, minute, daily, D, dates)


def test_intraday_peek_caught_in_strict_mode(data):
    minute, daily, dates = data
    with pytest.raises(LeakageError):
        check_op(cheat_closing_auction, {}, minute, daily, D, dates, apply_cutoff=False)


def test_production_cutoff_hides_bars_after_decision(data):
    minute, daily, dates = data
    out = cheat_closing_auction(ops.prepare_minute(ops.cutoff(minute, D)), daily, D)
    assert out["peek_auction"].null_count() == out.height
    assert ops.cutoff(minute, D)["bidx"].max() == D


def test_feature_cache_refuses_cheating_op(sandbox):
    ops.REGISTRY["_cheat_next_open"] = (cheat_next_day_open, "1")
    try:
        fc = FeatureCache("tiny")
        with pytest.raises(LeakageError):
            fc.build([{"op": "_cheat_next_open"}], date(2018, 7, 1), date(2018, 7, 31))
        assert not list((fc.root).glob("_cheat_next_open-*/*.parquet"))
    finally:
        ops.REGISTRY.pop("_cheat_next_open")


def test_cheat_would_have_looked_good(data):
    """Sanity check of the simulator: the peeking feature really predicts the label,
    so the test above is catching something that matters."""
    minute, daily, _ = data
    peek = cheat_next_day_open(ops.prepare_minute(minute), daily, D)
    nxt = daily.sort("symbol", "date").with_columns(
        (pl.col("open").shift(-1).over("symbol") * pl.col("adj_factor").shift(-1).over("symbol")
         / (pl.col("close") * pl.col("adj_factor")) - 1).alias("on")
    )
    j = peek.join(nxt.select("symbol", "date", "on"), on=["symbol", "date"]).drop_nulls()
    assert j.select(pl.corr("peek", "on", method="spearman")).item() > 0.3
