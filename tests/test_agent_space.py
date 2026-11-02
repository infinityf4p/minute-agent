from __future__ import annotations

import random

import pytest

from minute_agent.agent import space
from minute_agent.agent.loop import compact
from minute_agent.experiment import config as C


def test_samples_are_valid_and_diverse():
    rng = random.Random(0)
    cfgs = [space.sample(rng, seq_models=True) for _ in range(200)]
    for c in cfgs:
        space.check(c, seq_models=True)
        C.validate({"market": "cn", "dataset": "x", "eval_split": "valid", **c})
    hashes = {C.config_hash(c) for c in cfgs}
    assert len(hashes) == 200
    assert {c["model"]["type"] for c in cfgs} == set(space.models(True))
    assert compact(cfgs[0]).startswith("target=")


@pytest.mark.parametrize("patch", [
    {"target": "intraday"},
    {"features": [{"op": "ret", "window_min": [7]}]},
    {"features": [{"op": "nope"}]},
    {"model": {"type": "gru_seq"}},                 # sequence models off by default
    {"features": [{"op": "custom", "name": "x", "code": "def feature(m, daily, d): return m"}]},  # v2 off
    {"train": {"window_years": 9, "retrain_every": "quarter"}},
    {"eval_split": "test"},
])
def test_check_rejects(patch):
    base = space.sample(random.Random(1))
    with pytest.raises(space.SpaceError):
        space.check({**base, **patch})


def test_describe_mentions_every_op():
    text = space.describe(seq_models=True, custom_features=True)
    for op in space.OPS:
        assert op in text
    assert "gru_seq" in text and "custom" in text
