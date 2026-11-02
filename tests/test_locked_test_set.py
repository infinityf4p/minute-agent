"""The locked test period (2024-01 .. 2026-10) must be unreachable without the explicit switch."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from minute_agent import split
from minute_agent.data import store
from minute_agent.evaluator import evaluate
from minute_agent.experiment import config as C
from minute_agent.market import load_market


@pytest.mark.parametrize("reader", ["minute", "daily", "calendar"])
def test_reads_into_locked_period_raise(sandbox, reader):
    fn = {"minute": store.read_minute, "daily": store.read_daily, "calendar": store.read_calendar}[reader]
    with pytest.raises(split.LockedDataError):
        fn("edge", date(2023, 12, 1), date(2024, 1, 10))
    with pytest.raises(split.LockedDataError):
        fn("edge", date(2023, 12, 1), None)


def test_reads_before_lock_work_and_stop_at_boundary(sandbox):
    m = store.read_minute("edge", date(2023, 12, 1), date(2023, 12, 31))
    assert m.height > 0 and m["date"].max() <= date(2023, 12, 31)


def test_locked_partitions_are_not_listed(sandbox):
    months = store.minute_months("edge")
    assert months and all(date(y, m, 1) <= split.LAST_OPEN_DATE for y, m in months)


def test_security_table_hides_future(sandbox):
    sec = store.read_security("edge")
    assert sec.filter(pl.col("delist_date") > split.LAST_OPEN_DATE).height == 0
    assert sec.filter(pl.col("list_date") > split.LAST_OPEN_DATE).height == 0


def test_env_var_alone_does_not_unlock(sandbox, monkeypatch):
    monkeypatch.setenv(split.UNLOCK_ENV, split.UNLOCK_PHRASE)
    with pytest.raises(split.LockedDataError):
        store.read_daily("edge", date(2024, 1, 1), date(2024, 1, 31))


def test_flag_alone_does_not_unlock(sandbox):
    with pytest.raises(split.LockedDataError):
        store.read_daily("edge", date(2024, 1, 1), date(2024, 1, 31), allow_locked=True)


def test_env_var_plus_flag_unlocks(sandbox, monkeypatch):
    monkeypatch.setenv(split.UNLOCK_ENV, split.UNLOCK_PHRASE)
    d = store.read_daily("edge", date(2024, 1, 1), date(2024, 1, 31), allow_locked=True)
    assert d.height > 0


def test_config_cannot_target_test_split():
    cfg = {"target": "overnight", "eval_split": "test", "features": [{"op": "ret"}], "model": {"type": "lgbm"}}
    with pytest.raises(C.ConfigError):
        C.validate(cfg)


def test_evaluator_refuses_locked_dates():
    panel = pl.DataFrame({
        "date": [date(2024, 1, 2)] * 30, "symbol": [f"s{i}" for i in range(30)],
        "score": list(range(30)), "label": [0.0] * 30, "buyable": [True] * 30,
    })
    with pytest.raises(split.LockedDataError):
        evaluate(panel, load_market("cn"), "overnight")
