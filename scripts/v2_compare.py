"""Compare search arms at equal budget on the validation period (2022-2023, never shown to any arm).

    conda run -n minute-agent python scripts/v2_compare.py runs/v2_validation_b50_new.json runs/v2_validation_b50_old.json

Per arm and template: the validation objective of each session's finalist (selected by the same evaluator rule),
and per session the best finalist and the mean over its templates.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

TPLS = ["T1", "T2", "T3", "T4"]


def main():
    val = {}
    for p in sys.argv[1:]:
        val.update(json.load(open(p, encoding="utf-8")))
    fin = defaultdict(lambda: defaultdict(list))     # arm -> template -> [valid objective]
    rows = []
    for s, d in sorted(val.items()):
        arm = d.get("arm") or s.split("-")[1]
        per = {}
        for t, x in d["templates"].items():
            c = next(c for c in x["candidates"] if c["valid_run"] == x["finalist"])
            per[t] = c
            fin[arm][t].append(c["valid_objective"])
        rows.append((s, arm, per))
    print("Validation objective of each session's finalist (T1/T3/T4: Sharpe, T2: bp/day; x2 = at 2x costs)\n")
    print(f"{'session':22s} " + " ".join(f"{t:>14s}" for t in TPLS))
    for s, arm, per in rows:
        cells = []
        for t in TPLS:
            c = per.get(t)
            cells.append(f"{c['valid_objective']:+6.2f} ({c['valid_x2_objective']:+5.2f})" if c else f"{'-':>14s}")
        print(f"{s:22s} " + " ".join(f"{x:>14s}" for x in cells))
    print("\nMean over sessions (number of sessions with a finalist):")
    for arm in sorted(fin):
        print(f"  {arm:8s} " + "  ".join(
            f"{t}: {np.mean(fin[arm][t]):+.2f} (n={len(fin[arm][t])})" if fin[arm][t] else f"{t}: -" for t in TPLS))
    print("\nT2 finalists positive at 2x costs, and T1/T3/T4 finalists with validation Sharpe > 0.5 at 2x costs:")
    for arm in sorted(fin):
        good = sum(1 for s, a, per in rows if a == arm for t, c in per.items()
                   if (t == "T2" and c["valid_x2_objective"] > 0) or (t != "T2" and c["valid_x2_objective"] > 0.5))
        n = sum(1 for s, a, per in rows if a == arm)
        print(f"  {arm:8s} {good} usable finalists over {n} sessions")


if __name__ == "__main__":
    main()
