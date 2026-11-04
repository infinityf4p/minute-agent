import random

from minute_agent.agent import loop_v2
from minute_agent.pipeline import spec


class Stub:
    def __init__(self, rows):
        self.settings = spec.Settings(position_sizing=True, allow_t0=True)
        self._rows = rows

    def rows(self):
        return self._rows


def row(h, split, tpl, f1=None, f2=None, status="done"):
    return {"config_hash": h, "eval_split": split, "status": status, "record": {"cfg": {"template": tpl}},
            "metrics": {"fold1_objective": f1, "fold2_objective": f2}}


def test_success_needs_both_folds_positive():
    s = Stub([row("a", "screen", "T1"), row("a", "folds", "T1", 0.5, 0.2),
              row("b", "screen", "T1"), row("b", "folds", "T1", 1.0, -0.1),
              row("c", "screen", "T3")])
    st = loop_v2.template_stats(s)
    assert st["T1"] == [1, 1] and st["T3"] == [0, 1] and st["T2"] == [0, 0]


def test_untried_templates_first_then_sampling():
    rng = random.Random(0)
    s = Stub([row("a", "screen", "T1"), row("b", "screen", "T1")])
    assert loop_v2.pick_template(s, rng)[0] in {"T2", "T3", "T4"}
    rows = []
    for i, t in enumerate(["T1", "T2", "T3", "T4"] * 2):
        rows.append(row(f"{t}{i}", "screen", t))
        if t == "T2":
            rows.append(row(f"{t}{i}", "folds", t, 1.0, 1.0))
    picks = [loop_v2.pick_template(Stub(rows), rng)[0] for _ in range(200)]
    assert picks.count("T2") > 100
