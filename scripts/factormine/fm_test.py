"""Primary-metric test of a round: per session, found factors whose validation residual IC keeps the mining sign
with |t| >= 2; one-sided Mann-Whitney of each LLM arm against random, Holm-adjusted over the LLM arms.

    conda run --no-capture-output -n minute-agent python scripts/factormine/fm_test.py <validation json>
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import mannwhitneyu


def primary(s: dict) -> int:
    c = 0
    for m in s["members"]:
        sg, t = np.sign(m["mine"]["res_ic"]), m["valid"].get("res_t")
        if t is not None and np.isfinite(t) and np.sign(t) == sg and abs(t) >= 2:
            c += 1
    return c


def main():
    v = json.load(open(sys.argv[1]))
    arms: dict[str, list[tuple[str, int, float | None]]] = {}
    for name, s in v.items():
        va, vs = s.get("valid_all") or {}, s.get("valid_seeds") or {}
        inc = va["res_ic"] - vs["res_ic"] if va.get("res_ic") is not None and vs.get("res_ic") is not None else None
        arms.setdefault(s["arm"], []).append((name, primary(s), inc))
    for a, xs in arms.items():
        print(f"{a:9s} primary {[x[1] for x in xs]} mean {np.mean([x[1] for x in xs]):.1f} | composite increment "
              f"{[round(x[2], 4) if x[2] is not None else None for x in xs]}")
    r = [x[1] for x in arms.get("random", [])]
    llm = [a for a in arms if a != "random"]
    ps = {a: mannwhitneyu([x[1] for x in arms[a]], r, alternative="greater").pvalue for a in llm}
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, a in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[a]))
        adj[a] = run
    for a in llm:
        ri = [x[2] for x in arms["random"] if x[2] is not None]
        ai = [x[2] for x in arms[a] if x[2] is not None]
        p2 = mannwhitneyu(ai, ri, alternative="greater").pvalue if ai and ri else float("nan")
        print(f"{a} vs random: primary one-sided MW p = {ps[a]:.4f}, Holm-adjusted {adj[a]:.4f}; "
              f"composite increment p = {p2:.4f}")


if __name__ == "__main__":
    main()
