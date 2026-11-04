"""Run factor-mining sessions of one arm, one seed after another (resumes a session that already has rows).

    conda run --no-capture-output -n minute-agent python scripts/factormine/fm_run.py random --seeds 0 1 2 --budget 60
"""

from __future__ import annotations

import argparse

from minute_agent.factormine import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=search.ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--cfg", default="pilot")
    ap.add_argument("--tag", default="", help="suffix of the arm in the session name, e.g. -m")
    a = ap.parse_args()
    for seed in a.seeds:
        s = search.run(a.arm, seed, a.budget, a.cfg, a.tag)
        print(f"done {s.name}: {s.n()} candidates, {len(s.lib.found())} factors found", flush=True)


if __name__ == "__main__":
    main()
