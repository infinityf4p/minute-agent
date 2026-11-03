import random

import pytest

from minute_agent.agent.space import SpaceError
from minute_agent.pipeline import spec


def test_malformed_proposals_are_space_errors():
    s = spec.Settings(position_sizing=True, allow_t0=True)
    cfg = spec.sample(random.Random(0), s)
    spec.check(cfg, s)
    with pytest.raises(SpaceError):
        spec.check({**cfg, "features": ["ret_5", "rv_30"]}, s)
    with pytest.raises(SpaceError):
        spec.check("not a config", s)
