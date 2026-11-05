import random
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from minute_agent.factormine import dsl
from minute_agent.factormine.data import DAILY_VARS
from minute_agent.factormine.library import Library


def _bars(n_sym=3, n_days=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sym):
        for k in range(n_days):
            px = 10 * np.exp(np.cumsum(rng.normal(0, 0.002, 241)))
            for b in range(241):
                rows.append({"symbol": f"S{s}", "date": date(2020, 1, 2) + timedelta(days=k), "bidx": b,
                             "open": px[b], "high": px[b] * 1.001, "low": px[b] * 0.999, "close": px[b],
                             "volume": float(rng.integers(100, 1000)), "amount": float(px[b] * 500)})
    return pl.DataFrame(rows).with_columns(pl.col("bidx").cast(pl.Int16))


def test_random_factors_are_valid():
    rng = random.Random(0)
    for _ in range(200):
        f = dsl.random_factor(rng, DAILY_VARS)
        for n in f["intraday"].values():
            assert dsl.check_node(n) == n
        e = dsl.check_expr(f["combine"], set(f["intraday"]) | set(DAILY_VARS))
        assert dsl.vars_used(e) & set(f["intraday"])


@pytest.mark.parametrize("bad", [
    {"agg": "corr", "x": "ret", "win": "all", "cond": None},                     # corr without y
    {"agg": "share", "x": "ret", "win": "all", "cond": None},                     # share of a signed field
    {"agg": "mean", "x": "ret", "win": "auction", "cond": None},                 # auction has no return
    {"agg": "mean", "x": "close", "win": "all", "cond": None},                   # unknown field
])
def test_bad_nodes_rejected(bad):
    with pytest.raises(dsl.DSLError):
        dsl.check_node(bad)


def test_combine_limits():
    v = {"a"}
    with pytest.raises(dsl.DSLError):
        dsl.check_expr({"op": "delta", "x": {"var": "a"}, "n": 60}, v)
    with pytest.raises(dsl.DSLError):
        dsl.check_expr({"var": "label"}, v)
    deep = {"var": "a"}
    for _ in range(dsl.MAX_DEPTH + 1):
        deep = {"op": "neg", "x": deep}
    with pytest.raises(dsl.DSLError):
        dsl.check_expr(deep, v)


def test_intraday_nodes_ignore_bars_after_decision():
    m = _bars()
    d = 230
    rng = random.Random(1)
    nodes = {f"n{i}": dsl.random_node(rng) for i in range(30)}
    full = dsl.eval_nodes(dsl.prepare_bars(m, d), nodes, d).sort("symbol", "date")
    later = m.with_columns(pl.when(pl.col("bidx") > d).then(pl.col("close") * 3).otherwise(pl.col("close")).alias("close"),
                           pl.when(pl.col("bidx") > d).then(pl.col("volume") * 50).otherwise(pl.col("volume")).alias("volume"))
    changed = dsl.eval_nodes(dsl.prepare_bars(later, d), nodes, d).sort("symbol", "date")
    for c in nodes:
        a, b = full[c].to_numpy(), changed[c].to_numpy()
        assert np.allclose(a, b, equal_nan=True), c


def test_ts_ops_look_backwards_only():
    fr = pl.DataFrame({"symbol": ["A"] * 10, "date": [date(2020, 1, i + 1) for i in range(10)],
                       "a": [float(i) for i in range(10)]})
    for op in ["ts_mean", "ts_max", "delta", "ts_zscore"]:
        out1 = dsl.evaluate(fr, {"op": op, "x": {"var": "a"}, "n": 5})["f"].to_list()
        fr2 = fr.with_columns(pl.when(pl.col("date") > date(2020, 1, 6)).then(999.0).otherwise(pl.col("a")).alias("a"))
        out2 = dsl.evaluate(fr2, {"op": op, "x": {"var": "a"}, "n": 5})["f"].to_list()
        assert out1[:6] == out2[:6], op


def test_library_rules(tmp_path):
    lib = Library(tmp_path)
    base = {"coverage": 1.0, "res_ic": 0.02, "res_t": 6.0, "res_ic_folds": [0.02, 0.02]}
    assert lib.decide(base, {})[0] == "admitted"
    assert lib.decide({**base, "res_t": 2.0}, {})[0] == "rejected_weak"
    assert lib.decide({**base, "res_ic_folds": [0.03, -0.01]}, {})[0] == "rejected_weak"
    lib.members = [{"id": "m1", "metrics": {"res_ic": 0.01}, "fixed": False},
                   {"id": "ref", "metrics": {"res_ic": 0.005}, "fixed": True}]
    assert lib.decide(base, {"m1": 0.8, "ref": 0.1}) == ("replaced", "m1", "replaces m1 (corr +0.80)")
    assert lib.decide(base, {"ref": 0.9})[0] == "rejected_redundant"          # fixed members are never replaced
    assert lib.decide(base, {"m1": 0.8, "ref": 0.6})[0] == "rejected_redundant"


def test_combine_accepts_op_var_spelling():
    from minute_agent.factormine import dsl
    assert dsl.check_expr({"op": "ts_mean", "x": {"op": "var", "x": "a"}, "n": 10}, {"a"}) == \
        {"op": "ts_mean", "x": {"var": "a"}, "n": 10}
