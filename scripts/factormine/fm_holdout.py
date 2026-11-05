"""factormine's once-only holdout (2024-01 .. 2026-10) for the pre-registered sessions (confirmation round).

Same measurements as fm_validate on the holdout period; the orientation of every factor is its mining sign.
Needs MINUTE_AGENT_UNLOCK_TEST (see minute_agent.split) and reads only inside split.final_test_unlock, which logs
the opening to runs/locked_test_openings.jsonl. Refuses to run again once the result file is complete.
Holdout caches go to cache/factormine/<cfg>_holdout.

    MINUTE_AGENT_UNLOCK_TEST=... conda run --no-capture-output -n minute-agent python scripts/factormine/fm_holdout.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fm_validate import composite_metrics, oriented  # noqa: E402

from minute_agent import split  # noqa: E402
from minute_agent.factormine import config, data, evaluate  # noqa: E402
from minute_agent.factormine.library import Library  # noqa: E402

# Pre-registered (docs/factormine/confirmation.md): the confirmation-round sessions, code frozen at 427eff2.
SESSIONS = [f"pilot-{a}-c-s{s}" for a in ("random", "llm_code", "llm_dsl") for s in (8, 9, 10, 11, 12)]
OUT = config.runs_dir() / "holdout_result.json"


def _clean(v):
    return None if isinstance(v, float) and not np.isfinite(v) else str(v)


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {"sessions": {}}
    if res.get("complete"):
        raise SystemExit(f"holdout already evaluated ({OUT}); it is opened only once")
    with split.final_test_unlock("factormine holdout 2024-01..2026-10, confirmation-round sessions (once only)"):
        res.setdefault("opened_at", datetime.now().isoformat(timespec="seconds"))
        for s in SESSIONS:
            if s in res["sessions"]:
                continue
            sd = config.runs_dir() / "sessions" / s
            meta = json.loads((sd / "session.json").read_text())
            base = config.get(meta["cfg"])
            cfg = config.holdout_cfg(base)
            assert data.universe(cfg) == data.universe(base), "holdout universe differs"
            H = cfg["holdout"]
            lib = Library(sd / "library")
            members, cols = [], {}
            for x in lib.members:
                sign = float(x["metrics"]["sign"])
                cols[x["id"]] = oriented(cfg, x["spec"], H, sign)
                if x.get("fixed"):
                    continue
                mh, _ = evaluate.metrics(evaluate.factor_frame(cfg, x["spec"], H), H, sign=sign)
                members.append({"id": x["id"], "mechanism": x["spec"].get("mechanism"), "mine": x["metrics"],
                                "holdout": {k: mh.get(k) for k in ["coverage", "res_ic", "res_t", "res_ic_folds",
                                                                    "ic", "rank_autocorr", "ls_gross_bp",
                                                                    "ls_net_bp"]}})
                print(s, x["id"], f"mine {x['metrics']['res_ic']:+.4f} -> holdout {mh.get('res_ic', float('nan')):+.4f}"
                      f" (t {mh.get('res_t', float('nan')):+.1f})", flush=True)
            seeds = [x["id"] for x in lib.members if x.get("fixed")]
            found = [x["id"] for x in lib.members if not x.get("fixed")]
            out = {"arm": meta["arm"], "seed": meta["seed"], "n_found": len(found), "members": members}
            for tag, ids in (("seeds", seeds), ("found", found), ("all", seeds + found)):
                out[f"holdout_{tag}"] = composite_metrics(cfg, {k: cols[k] for k in ids}, H)
            print(s, "composite holdout res_ic: seeds", f"{out['holdout_seeds'].get('res_ic', float('nan')):+.4f}",
                  "all", f"{out['holdout_all'].get('res_ic', float('nan')):+.4f}", flush=True)
            res["sessions"][s] = out
            OUT.write_text(json.dumps(res, indent=1, default=_clean))
        res["complete"] = True
        OUT.write_text(json.dumps(res, indent=1, default=_clean))
    print("saved", OUT)


if __name__ == "__main__":
    main()
