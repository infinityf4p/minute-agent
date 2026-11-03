from __future__ import annotations

import random
from datetime import date

import polars as pl
import pytest

from minute_agent.agent.space import OPS, _subset
from minute_agent.features.atoms import AtomCache, atomize
from minute_agent.features.cache import FeatureCache


def _random_specs(rng, k=5):
    specs = []
    for op in rng.sample(sorted(OPS), k):
        s = {"op": op}
        for p, (kind, dom) in OPS[op].items():
            s[p] = rng.choice(dom) if kind == "choice" else _subset(rng, dom)
        specs.append(s)
    return specs


def test_atomize_daily_lag_is_a_product():
    a = atomize({"op": "daily_lag", "fields": ["ret", "turnover"], "lags": [1, 5]})
    assert len(a) == 4 and {"op": "daily_lag", "fields": ["turnover"], "lags": [5]} in a


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_atoms_equal_whole_specs(sandbox, monkeypatch, tmp_path, seed):
    monkeypatch.setenv("MINUTE_AGENT_CACHE", str(tmp_path / f"c{seed}"))
    specs = _random_specs(random.Random(seed))
    start, end = date(2018, 7, 1), date(2018, 9, 30)
    fc = FeatureCache("tiny")
    fc.build(specs, start, end)
    whole = fc.load(specs, start, end)
    ac = AtomCache("tiny")
    ac.build(specs, start, end)
    atoms = ac.load(specs, start, end)
    assert sorted(whole.columns) == sorted(atoms.columns)
    a = atoms.select(whole.columns)
    assert whole.height == a.height
    for c in whole.columns:
        x, y = whole[c], a[c]
        if x.dtype in (pl.Float64, pl.Float32):
            assert ((x - y).abs().fill_null(0.0).fill_nan(0.0).max() or 0.0) < 1e-12, c
            assert (x.is_null() == y.is_null()).all(), c
        else:
            assert x.equals(y), c


def test_atom_subset_reuses_cache(sandbox, monkeypatch, tmp_path):
    monkeypatch.setenv("MINUTE_AGENT_CACHE", str(tmp_path / "r"))
    start, end = date(2018, 7, 1), date(2018, 7, 31)
    ac = AtomCache("tiny")
    ac.build([{"op": "ret", "window_min": [5, 30, 60]}], start, end)
    st = ac.build([{"op": "ret", "window_min": [30, 5]}], start, end)
    assert st["months_built"] == 0                     # every atom already cached
    x = ac.load([{"op": "ret", "window_min": [30]}], start, end)
    assert "ret_30" in x.columns and "ret_5" not in x.columns
