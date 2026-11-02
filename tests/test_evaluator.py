from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from minute_agent.evaluator import evaluate
from minute_agent.evaluator.integrity import verify
from minute_agent.experiment.ledger import Ledger
from minute_agent.market import load_market


def _panel(n_days=60, n=200, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = date(2022, 3, 1)
    for k in range(n_days):
        s = rng.normal(size=n)
        lab = signal * s * 0.01 + rng.normal(0, 0.01, n)
        rows.append(pl.DataFrame({
            "date": [d0 + timedelta(days=k)] * n, "symbol": [f"s{i}" for i in range(n)],
            "score": s, "label": lab, "buyable": [True] * n,
        }))
    return pl.concat(rows)


def test_evaluator_is_frozen():
    assert len(verify()) == 16


def test_modified_evaluator_is_detected(monkeypatch):
    from minute_agent.evaluator import integrity

    real = integrity.current()
    tampered = {**real, "core.py": "0" * 64}
    monkeypatch.setattr(integrity, "current", lambda: tampered)
    with pytest.raises(integrity.EvaluatorModifiedError):
        integrity.verify()


def test_perfect_signal():
    p = _panel().with_columns(pl.col("label").alias("score"))
    m = evaluate(p, load_market("cn"), "overnight")
    assert m["ic"] == pytest.approx(1.0) and m["rank_ic"] == pytest.approx(1.0)
    assert m["group_ret_bp"] == sorted(m["group_ret_bp"])
    assert m["excess_gross_bp"] > 0


def test_random_signal_has_no_ic():
    m = evaluate(_panel(signal=0.0), load_market("cn"), "overnight")
    assert abs(m["ic"]) < 0.02


def test_overnight_pays_full_round_trip():
    rules = load_market("cn")
    m = evaluate(_panel(signal=0.5), rules, "overnight")
    expected = rules.buy_cost(date(2022, 3, 1)) + rules.sell_cost(date(2022, 3, 1))
    assert m["cost_bp"] == pytest.approx(expected * 1e4)
    assert m["excess_net_bp"] == pytest.approx(m["excess_gross_bp"] - m["cost_bp"])


def test_stamp_duty_schedule():
    rules = load_market("cn")
    assert rules.sell_cost(date(2023, 8, 25)) - rules.sell_cost(date(2023, 8, 28)) == pytest.approx(0.0005)


def test_unbuyable_names_excluded_from_long_leg():
    p = _panel(signal=1.0).with_columns((pl.col("score") < 1.0).alias("buyable"))
    m = evaluate(p, load_market("cn"), "overnight")
    assert m["unbuyable_frac"] > 0.1
    assert m["avg_top_names"] < 200 / 10 * 0.95


def test_ledger_is_append_only(tmp_path):
    lg = Ledger(tmp_path / "l.sqlite")
    lg.start("r1", {"target": "overnight"}, "h", {"source": "test"})
    lg.finish("r1", {"ic": 0.1}, {"total": 1.0})
    with pytest.raises(sqlite3.DatabaseError):
        with sqlite3.connect(lg.path) as c:
            c.execute("DELETE FROM runs")
    with pytest.raises(sqlite3.DatabaseError):
        with sqlite3.connect(lg.path) as c:
            c.execute("UPDATE runs SET ic = 0.9 WHERE run_id='r1'")
    assert lg.rows()[0]["ic"] == pytest.approx(0.1)
