from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from minute_agent.data import store
from minute_agent.features import ops
from minute_agent.features.extra_ops import CustomFeatureError, op_custom, seq_shape, static_check
from minute_agent.features.leakage import LeakageError, check_op, pick_check_dates

D = 230

HONEST = '''
import polars as pl
def feature(m, daily, d):
    m = m.filter(pl.col("bidx") <= d)
    late = pl.col("bidx") > d - 20
    return m.group_by("symbol", "date").agg(
        (pl.col("volume").filter(late).sum() / pl.col("volume").sum()).alias("late_share"),
        pl.col("lr").filter(late).sum().alias("late_ret"),
    )
'''
CHEAT_NEXT_DAY = '''
import polars as pl
def feature(m, daily, d):
    c = m.group_by("symbol", "date").agg(pl.col("close").last().alias("c")).sort("symbol", "date")
    return c.with_columns((pl.col("c").shift(-1).over("symbol") / pl.col("c")).alias("peek")).drop("c")
'''
CHEAT_INTRADAY = '''
import polars as pl
def feature(m, daily, d):
    return m.group_by("symbol", "date").agg(pl.col("close").filter(pl.col("bidx") == 240).last().alias("x"))
'''


@pytest.fixture(scope="module")
def data(sandbox):
    start, end = date(2018, 8, 1), date(2018, 8, 31)
    minute = store.read_minute("tiny", start, end)
    daily = store.read_daily("tiny", start - timedelta(days=120), end)
    return minute, daily, pick_check_dates(minute["date"].unique().to_list(), k=3)


@pytest.mark.parametrize("code", [
    "import os\ndef feature(m, daily, d):\n    return m",
    "def feature(m, daily, d):\n    open('x')\n    return m",
    "import polars as pl\ndef feature(m, daily, d):\n    return pl.read_csv('x')",
    "def feature(m, daily, d):\n    return m.__class__",
    "def feature(m, daily, d):\n    return eval('1')",
    "def other(m, daily, d):\n    return m",
])
def test_static_check_rejects(code):
    with pytest.raises(CustomFeatureError):
        static_check(code)


def test_honest_custom_feature_passes(data):
    minute, daily, dates = data
    check_op(op_custom, {"name": "late", "code": HONEST}, minute, daily, D, dates)
    check_op(op_custom, {"name": "late", "code": HONEST}, minute, daily, D, dates, apply_cutoff=False)


def test_custom_cross_day_cheat_caught(data):
    minute, daily, dates = data
    with pytest.raises(LeakageError):
        check_op(op_custom, {"name": "peek", "code": CHEAT_NEXT_DAY}, minute, daily, D, dates)


def test_custom_intraday_cheat_caught_in_strict_mode(data):
    minute, daily, dates = data
    with pytest.raises(LeakageError):
        check_op(op_custom, {"name": "peek", "code": CHEAT_INTRADAY}, minute, daily, D, dates, apply_cutoff=False)


def test_seq_bars_shape(data):
    minute, daily, _ = data
    out = ops.compute("seq_bars", ops.prepare_minute(ops.cutoff(minute, D)), daily, D, {"bucket_min": 10})
    T = seq_shape(D, 10)
    assert T == 24
    assert len([c for c in out.columns if c.startswith("sq_")]) == T * 6
    assert out.select(pl.all().exclude("symbol", "date").is_null().sum()).sum_horizontal().item() == 0
