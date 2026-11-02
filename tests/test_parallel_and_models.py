from __future__ import annotations

import os
from datetime import date

import numpy as np
import polars as pl
import pytest

from minute_agent.features.cache import FeatureCache
from minute_agent.models import cuda_available, make_model

SPECS = [{"op": "ret", "window_min": [5, 30]}, {"op": "volume_profile"},
         {"op": "daily_lag", "fields": ["ret"], "lags": [1, 5]}]


def test_parallel_feature_build_matches_sequential(sandbox, monkeypatch, tmp_path):
    start, end = date(2018, 7, 1), date(2018, 9, 30)
    monkeypatch.setenv("MINUTE_AGENT_CACHE", str(tmp_path / "seq"))
    seq_fc = FeatureCache("tiny")
    seq_fc.build(SPECS, start, end, workers=1)
    seq = seq_fc.load(SPECS, start, end)
    monkeypatch.setenv("MINUTE_AGENT_CACHE", str(tmp_path / "par"))
    par_fc = FeatureCache("tiny")
    st = par_fc.build(SPECS, start, end, workers=3)
    par = par_fc.load(SPECS, start, end)
    assert st.months_built == 3
    assert seq.equals(par)


@pytest.mark.parametrize("spec", [
    {"type": "lgbm", "params": {"n_estimators": 20}},
    {"type": "xgb", "params": {"n_estimators": 20, "device": "cpu"}},
    {"type": "mlp", "params": {"epochs": 2, "device": "cpu"}},
    {"type": "linear"},
])
def test_models_learn_a_linear_signal(spec):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 5))
    y = X[:, 0] * 0.5 + rng.normal(0, 1, 20000)
    m = make_model(spec)
    m.set_threads(2)
    m.fit(X[:15000], y[:15000], [f"f{i}" for i in range(5)])
    p = m.predict(X[15000:])
    assert np.corrcoef(p, y[15000:])[0, 1] > 0.3


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("t", ["xgb", "mlp"])
def test_gpu_models(t):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 5))
    y = X[:, 0] + rng.normal(0, 1, 20000)
    m = make_model({"type": t, "params": {"device": "cuda", **({"epochs": 2} if t == "mlp" else {"n_estimators": 20})}})
    assert m.device == "cuda"
    m.fit(X, y, [f"f{i}" for i in range(5)])
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.3
