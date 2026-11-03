"""Evaluator-only final selection on the validation period (2022-2023) for v2 sessions.

Rule (fixed before any validation result is seen):
  1. per session and template, take the top-K configs by two-fold (training-period) objective
  2. run each on the validation period (phase "valid"); the agent never sees these results
  3. per session and template, the finalist is the candidate with the best validation objective
  4. deflated Sharpe of each finalist's validation series, with N = all evaluations of that session
     and the cross-trial variance of Sharpe estimated from that session's fold-phase daily series

    conda run -n minute-agent python scripts/v2_validate.py <session> [<session> ...] [--k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minute_agent.experiment.ledger import Ledger  # noqa: E402
from minute_agent.paths import runs_root  # noqa: E402
from minute_agent.pipeline import spec  # noqa: E402
from minute_agent.pipeline.run import run_pipeline  # noqa: E402
from minute_agent.report import stats  # noqa: E402


def daily_ret(run_id: str) -> np.ndarray | None:
    p = runs_root() / run_id / "daily.parquet"
    return pl.read_parquet(p)["ret"].to_numpy() if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    ledger = Ledger()
    for s in a.sessions:
        rows = ledger.query("SELECT * FROM runs WHERE session=?", (s,))
        n_trials = sum(1 for r in rows if not (r["status"] == "failed" and (r["error"] or "").startswith("interrupted")))
        meta = json.loads((runs_root() / "sessions" / s / "session.json").read_text())
        settings = spec.Settings(**meta["settings"])
        folds = [r for r in rows if r["eval_split"] == "folds" and r["status"] == "done" and r["objective"] is not None]
        srs = [stats.sharpe(x) for r in folds if (x := daily_ret(r["run_id"])) is not None and len(x) > 20]
        var_sr = float(np.nanvar(srs, ddof=1)) if len(srs) > 2 else float("nan")
        by_t: dict[str, list] = {}
        for r in folds:
            by_t.setdefault(json.loads(r["config_json"])["template"], []).append(r)
        sess_out = {"n_trials": n_trials, "var_sr_daily": var_sr, "templates": {}}
        for tpl, rs in by_t.items():
            cands = sorted(rs, key=lambda r: -r["objective"])[: a.k]
            evals = []
            for r in cands:
                cfg = json.loads(r["config_json"])["cfg"]
                res = run_pipeline(cfg, settings, phase="valid", source="evaluator", session=f"{s}::valid",
                                   ledger=ledger, verbose=True)
                x = daily_ret(res["run_id"])
                evals.append({"folds_run": r["run_id"], "folds_objective": r["objective"], "valid_run": res["run_id"],
                              "valid_objective": res["metrics"]["objective"], "valid_sharpe": res["metrics"]["sharpe"],
                              "valid_bp_per_day": res["metrics"]["mean_bp_per_day"],
                              "valid_x2_objective": res["metrics"]["x2_objective"],
                              "dsr": stats.deflated_sharpe(x, n_trials, var_sr) if x is not None and np.isfinite(var_sr) else None,
                              "config": cfg})
            best = max(evals, key=lambda e: e["valid_objective"] if np.isfinite(e["valid_objective"]) else -1e9)
            sess_out["templates"][tpl] = {"candidates": evals, "finalist": best["valid_run"]}
        print(json.dumps({t: {"finalist": v["finalist"]} for t, v in sess_out["templates"].items()}), flush=True)
        p = runs_root() / "v2_validation.json"   # save after every session
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev[s] = sess_out
        p.write_text(json.dumps(prev, indent=1, default=str))
        print("saved", s, p, flush=True)


if __name__ == "__main__":
    main()
