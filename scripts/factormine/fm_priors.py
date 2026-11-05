"""Freeze per-mechanism admission counts of finished LLM sessions (mining period only) as the scheduler prior.

    conda run --no-capture-output -n minute-agent python scripts/factormine/fm_priors.py <sessions...> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minute_agent.factormine import config, prompts
from minute_agent.factormine.library import Ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    led = Ledger(config.runs_dir() / "ledger.sqlite")
    st = {m: [0, 0] for m in prompts.MECHANISMS}
    for s in a.sessions:
        for r in led.rows(s):
            if r["mechanism"] in st:
                st[r["mechanism"]][0 if r["status"] in ("admitted", "replaced") else 1] += 1
    out = {"sessions": a.sessions, "counts": {m: {"admitted": x, "other": y} for m, (x, y) in st.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for m, (x, y) in sorted(st.items(), key=lambda kv: -kv[1][0] / max(1, sum(kv[1]))):
        print(f"{m:24s} {x:3d}/{x + y:3d}")


if __name__ == "__main__":
    main()
