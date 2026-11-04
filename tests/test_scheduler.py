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


def test_three_folds_min_and_prompt():
    r = {"metrics": {"fold1_objective": 0.4, "fold2_objective": 1.2, "fold3_objective": -0.1}}
    assert loop_v2._min_fold(r) == -0.1
    s = spec.Settings(position_sizing=True, allow_t0=True, custom_features=True)
    p = loop_v2.system_prompt(s, "folds3", robust=True)
    assert "three folds" in p and "2019" in p and "ALL three folds" in p and "CUSTOM FEATURES" in p
    assert "two folds" not in p
    p2 = loop_v2.system_prompt(spec.Settings(position_sizing=True, allow_t0=True), "folds", robust=False)
    assert "two folds" in p2 and "CUSTOM FEATURES" not in p2
