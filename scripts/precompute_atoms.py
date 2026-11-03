"""Precompute every atomic feature of the v2 action space (book decision times on the full market,
then T0 grid times on the base pool), most-used decision times first.

    conda run -n minute-agent python scripts/precompute_atoms.py [--workers 1]
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2023-12-31")
    a = ap.parse_args()
    from minute_agent.agent.space import OPS
    from minute_agent.features.atoms import AtomCache
    from minute_agent.pipeline import spec
    from minute_agent.pipeline.run import base_pool

    full = []
    for op, ps in OPS.items():
        s = {"op": op}
        for p, (kind, dom) in ps.items():
            s[p] = dom if kind == "subset" else None
        if any(v is None for v in s.values()):
            for p, (kind, dom) in ps.items():
                if kind == "choice":
                    for v in dom:
                        full.append({**{k: v2 for k, v2 in s.items() if v2 is not None}, p: v})
        else:
            full.append(s)
    use = Counter()
    try:
        c = sqlite3.connect("runs/ledger.sqlite")
        for (cj,) in c.execute("SELECT config_json FROM runs WHERE session LIKE 'p4-%'"):
            cfg = json.loads(cj).get("cfg", {})
            if "decision_time" in cfg:
                use[cfg["decision_time"]] += 1
    except Exception:
        pass
    times = sorted(spec.DECISION_TIMES, key=lambda t: -use[t])
    s0, e0 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    print("specs:", len(full), "book times:", times, flush=True)
    for t in times:
        t0 = time.time()
        r = AtomCache("cn", "cn", t).build(full, s0, e0, workers=a.workers)
        print(f"book {t}: {r} {time.time() - t0:.0f}s", flush=True)
    syms, tag = base_pool(spec.Settings(), "dev")
    grid = sorted({h for g in spec.T0_GRIDS.values() for h in g} | {spec.T0_OVERNIGHT_DECISION})
    for t in grid:
        t0 = time.time()
        r = AtomCache("cn", "cn", t, symbols=syms, universe_tag=tag).build(full, s0, e0, workers=a.workers)
        print(f"pool {t}: {r} {time.time() - t0:.0f}s", flush=True)
    print("PRECOMPUTE DONE", flush=True)


if __name__ == "__main__":
    main()
