from __future__ import annotations

from datetime import date

import polars as pl

from minute_agent import panel as P
from minute_agent.data import store
from minute_agent.data.schema import bar_index_at


def test_bar_layout():
    assert bar_index_at("cn", "09:30") == 0
    assert bar_index_at("cn", "11:30") == 120
    assert bar_index_at("cn", "13:01") == 121
    assert bar_index_at("cn", "12:00") == 120      # lunch: last bar is 11:30
    assert bar_index_at("cn", "14:50") == 230
    assert bar_index_at("cn", "15:00") == 240
    assert bar_index_at("us", "15:50") == 379


def test_sim_minute_shape(sandbox):
    m = store.read_minute("tiny", date(2018, 6, 1), date(2018, 10, 31))
    d = store.read_daily("tiny", date(2018, 6, 1), date(2018, 10, 31))
    per_day = m.group_by("symbol", "date").len()
    assert per_day["len"].unique().to_list() == [241]
    # suspended days have a daily row but no minute bars
    susp = d.filter(pl.col("is_suspended")).select("symbol", "date")
    assert susp.height > 0
    assert susp.join(per_day, on=["symbol", "date"], how="inner").height == 0
    # every bar respects the daily limits
    j = m.join(d.select("symbol", "date", "up_limit", "down_limit"), on=["symbol", "date"])
    assert j.filter(pl.col("high") > pl.col("up_limit") + 1e-9).height == 0
    assert j.filter(pl.col("low") < pl.col("down_limit") - 1e-9).height == 0


def test_close_auction_bars(sandbox):
    m = store.read_minute("tiny", date(2018, 9, 3), date(2018, 9, 7))
    late = m.filter(pl.col("bidx").is_in([238, 239]))
    assert late["volume"].sum() == 0          # 14:58 and 14:59 are empty in the closing auction


def test_adjusted_labels_have_no_ex_date_jumps(sandbox):
    """On ex-dates the unadjusted price drops; the adjusted label must not."""
    cal = store.read_calendar("tiny", date(2018, 6, 1), date(2018, 10, 31))
    d = store.read_daily("tiny", date(2018, 6, 1), date(2018, 10, 31))
    lab = P.labels(d, cal, "next_cc")
    ex = d.sort("symbol", "date").with_columns(
        (pl.col("adj_factor") != pl.col("adj_factor").shift(-1).over("symbol")).alias("ex_next")
    ).filter(pl.col("ex_next")).select("symbol", "date")
    assert ex.height > 0
    vals = ex.join(lab, on=["symbol", "date"]).drop_nulls()["label"]
    assert vals.abs().max() < 0.45
