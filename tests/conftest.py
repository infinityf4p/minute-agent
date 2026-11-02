from __future__ import annotations

from datetime import date

import pytest

from minute_agent.data import simulate


@pytest.fixture(scope="session")
def sandbox(tmp_path_factory):
    """Point data/cache/runs at a temp dir and generate two small simulated datasets:
    'tiny' (2018-06..2018-10) and 'edge' (2023-11..2024-02, crosses into the locked period)."""
    root = tmp_path_factory.mktemp("ma")
    mp = pytest.MonkeyPatch()
    mp.setenv("MINUTE_AGENT_DATA", str(root / "data"))
    mp.setenv("MINUTE_AGENT_CACHE", str(root / "cache"))
    mp.setenv("MINUTE_AGENT_RUNS", str(root / "runs"))
    mp.delenv("MINUTE_AGENT_UNLOCK_TEST", raising=False)
    simulate.generate("tiny", simulate.SimConfig(n_stocks=40, start=date(2018, 6, 1), end=date(2018, 10, 31),
                                                 p_suspend=0.01, frac_st=0.2, seed=1), verbose=False)
    simulate.generate("edge", simulate.SimConfig(n_stocks=20, start=date(2023, 11, 1), end=date(2024, 2, 29),
                                                 frac_delist=0.3, seed=2), verbose=False)
    yield root
    mp.undo()
