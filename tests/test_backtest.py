from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from minute_agent.backtest.book import Dense, simulate_book
from minute_agent.backtest.costs import CostModel
from minute_agent.backtest.t0 import simulate_t0

ZERO = CostModel({"name": "zero", "participation_cap": 1.0})


def test_stamp_duty_switch_and_min_commission():
    c = CostModel.load("default")
    before, after = c.round_trip_bps(date(2023, 8, 25)), c.round_trip_bps(date(2023, 8, 28))
    assert before - after == pytest.approx(5.0)          # 10bp -> 5bp on the sell side
    small = c.trade_cost(np.array([1000.0]), np.array([False]), np.array([np.datetime64("2024-01-02")]))
    assert small[0] >= 5.0                                 # minimum commission


def _dense(T=4, N=4, **over):
    dates = [date(2022, 3, 1) + timedelta(days=i) for i in range(T)]
    A = {k: np.ones((T, N)) * 10.0 for k in ("close", "open_px", "px")}
    A.update({"adj": np.ones((T, N)), "avail": np.full((T, N), 1e9), "can_buy": np.ones((T, N)),
              "can_sell": np.ones((T, N)), "open_can_sell": np.ones((T, N)), "sigma": np.full((T, N), 0.001),
              "in_univ": np.ones((T, N)), "score": np.tile(np.arange(N, dtype=float), (T, 1)),
              "univ_ret20": np.zeros((T, N))})
    A.update(over)
    return Dense(np.array(dates, dtype="datetime64[D]"), [f"s{i}" for i in range(N)], A)


def test_limit_up_at_decision_is_skipped():
    now = np.ones((4, 4))
    now[:, 3] = 0                                          # best-scored stock sits at the up limit at 14:50
    d = _dense(buyable_now=now, can_buy=now.copy())
    daily, pos = simulate_book(d, {"top_k": 1, "mode": "hold"}, ZERO)
    held = [p[1] for p in pos]
    assert all(v[3] == 0 for v in held)
    assert held[-1][2] > 0                                 # it picks the next best instead


def test_unfillable_order_stays_in_cash():
    can_buy = np.ones((4, 4))
    can_buy[:, 3] = 0                                      # buyable at the decision, locked in the next bars
    daily, pos = simulate_book(_dense(can_buy=can_buy), {"top_k": 1, "mode": "hold"}, ZERO)
    assert all(p[1].sum() == 0 for p in pos)               # no look-ahead switch to another stock
    assert daily["exposure"].max() == 0


def test_t_plus_one_blocks_same_day_stop_loss():
    px = np.ones((4, 4)) * 10.0
    close = np.ones((4, 4)) * 10.0
    close[0, 3] = 5.0                                      # bought at 10, closes at 5 on day 0
    px[1, 3] = 5.0
    close[1:, 3] = 5.0
    daily, pos = simulate_book(_dense(px=px, close=close, open_px=close.copy()),
                               {"top_k": 1, "mode": "hold", "stop_loss": 0.1, "rebalance_every": 100}, ZERO)
    assert pos[0][1][3] > 0                                # still held at the start of day 1 (no same-day sell)
    assert pos[1][1][3] == 0                               # stopped out on day 1 (age >= 1)


def test_participation_cap():
    d = _dense(avail=np.full((4, 4), 100.0))               # 100 shares x 10 CNY = 1000 CNY per bar
    cm = CostModel({"name": "cap", "participation_cap": 0.1})
    daily, pos = simulate_book(d, {"top_k": 1, "mode": "hold"}, cm, capital=1e6)
    assert pos[0][1][3] == pytest.approx(100.0)            # 10% of 1000 CNY


def test_next_open_mode_sells_at_open():
    open_px = np.ones((4, 4)) * 10.0
    open_px[1, 3] = 11.0                                   # +10% gap for the held stock
    daily, _ = simulate_book(_dense(open_px=open_px), {"top_k": 1, "mode": "next_open"}, ZERO, capital=1e6)
    assert daily["ret_net"][1] == pytest.approx(0.1, rel=1e-6)
    # the gap is earned against a universe that gapped +2.5% on average
    assert daily["excess_net"][1] == pytest.approx(0.1 - 0.025, rel=1e-6)


def test_holding_the_universe_has_zero_excess():
    rng = np.random.default_rng(0)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, (6, 4)), axis=0))
    d = _dense(T=6, close=close, open_px=close.copy(), px=close.copy())
    daily, _ = simulate_book(d, {"top_k": 4, "mode": "hold", "max_weight": 1.0}, ZERO)
    assert np.abs(daily["excess_net"].to_numpy()[1:]).max() < 1e-9


def _t0_frames(px_path, can_buy=None, next_open=10.0, next_can_buy=True):
    days = [date(2022, 3, 1), date(2022, 3, 2)]
    ds = [100, 160, 225]
    base = pl.DataFrame({"symbol": ["a", "a"], "date": days, "base_value": [1e6, 1e6]})
    sig = pl.DataFrame({"symbol": ["a"] * 3, "date": [days[0]] * 3, "d": ds, "score": [-1.0, 0.0, 0.0]},
                       schema_overrides={"d": pl.Int64})
    exe = pl.DataFrame({"symbol": ["a"] * 3, "date": [days[0]] * 3, "d": ds, "px": px_path,
                        "avail": [1e9] * 3, "can_buy": can_buy or [True] * 3, "can_sell": [True] * 3,
                        "sigma": [0.001] * 3}, schema_overrides={"d": pl.Int64})
    auc = pl.DataFrame({"symbol": ["a", "a"], "date": days, "open_px": [10.0, next_open],
                        "open_can_buy": [True, next_can_buy], "open_can_sell": [True, True], "close_px": [10.0, 10.0]})
    return base, sig, exe, auc


def test_t0_sell_first_then_forced_buyback():
    base, sig, exe, auc = _t0_frames([10.0, 9.5, 9.0])
    # single stock: with entry_q=1.0 the quantile thresholds equal the score itself
    daily, summ = simulate_t0(base, sig, exe, auc, {"direction": "sell_first", "size_frac": 0.5, "entry_q": 1.0},
                              ZERO)
    # sold 50k shares at 10, bought back at 9.0 at the flatten point -> +50,000 CNY on 1e6 base
    assert daily["net"][0] == pytest.approx(50_000.0)
    assert summ["round_trips"] == 1 and summ["carried_overnight_frac"] == 0


def test_t0_failed_buyback_settles_next_open():
    base, sig, exe, auc = _t0_frames([10.0, 10.5, 11.0], can_buy=[True, False, False], next_open=12.0)
    daily, summ = simulate_t0(base, sig, exe, auc, {"direction": "sell_first", "size_frac": 0.5, "entry_q": 1.0},
                              ZERO)
    # could not buy back on day 0 (limit-up), bought at next open 12.0: loss 50k shares x 2
    assert daily["net"].sum() == pytest.approx(-100_000.0)
    assert summ["carried_overnight_frac"] == 1.0


def test_unlisted_names_do_not_poison_the_book():
    close = np.ones((4, 4)) * 10.0
    close[:2, 0] = np.nan                                  # stock 0 lists on day 2
    d = _dense(close=close, open_px=close.copy(), px=close.copy())
    d.arrays["score"][:2, 0] = np.nan
    daily, _ = simulate_book(d, {"top_k": 2, "mode": "hold"}, CostModel.load("default"))
    assert np.isfinite(daily["equity"].to_numpy()).all()
    assert np.isfinite(daily["excess_net"].to_numpy()).all()


def test_t0_partial_flatten_is_retried():
    days = [date(2022, 3, 1), date(2022, 3, 2)]
    base = pl.DataFrame({"symbol": ["a", "a"], "date": days, "base_value": [1e6, 1e6]})
    ds = [100, 225, 230]
    sig = pl.DataFrame({"symbol": ["a"] * 3, "date": [days[0]] * 3, "d": ds, "score": [-1.0, None, None]},
                       schema_overrides={"d": pl.Int64, "score": pl.Float64})
    # 30,000 shares can trade per bar under a 100% cap; we sold 50,000 at d=100
    exe = pl.DataFrame({"symbol": ["a"] * 3, "date": [days[0]] * 3, "d": ds, "px": [10.0, 9.0, 9.0],
                        "avail": [1e9, 30_000.0, 30_000.0], "can_buy": [True] * 3, "can_sell": [True] * 3,
                        "sigma": [0.001] * 3}, schema_overrides={"d": pl.Int64})
    auc = pl.DataFrame({"symbol": ["a", "a"], "date": days, "open_px": [10.0, 10.0], "open_can_buy": [True, True],
                        "open_can_sell": [True, True], "close_px": [10.0, 10.0]})
    daily, summ = simulate_t0(base, sig, exe, auc, {"direction": "sell_first", "size_frac": 0.5, "entry_q": 1.0,
                                                    "flatten_at": 225}, ZERO)
    assert daily["net"].sum() == pytest.approx(50_000.0)   # 30k at 9 at 14:45, 20k at 9 at 14:50
    assert summ["round_trips"] == 1 and summ["carried_overnight_frac"] == 0
