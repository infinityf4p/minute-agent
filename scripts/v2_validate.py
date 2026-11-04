"""Evaluator-only final selection on the validation period (2022-2023) for v2 sessions.

Rule (fixed before any validation result is seen):
  1. per session and template, take the top-K configs by two-fold (training-period) objective
  2. run each on the validation period (phase "valid"); the agent never sees these results
  3. per session and template, the finalist is the candidate with the best validation objective
  4. deflated Sharpe of each candidate's validation series, per template (the templates' daily series are not
     comparable: a T2 overlay has a much smaller variance than a T1 book). N = distinct configs of that template
     in the session; the cross-trial variance of the daily Sharpe comes from the screen-phase series of that
     template pooled over all P4 sessions (all trials, not only the promoted ones; one session alone may have
     too few trials of a template).

    --dsr-only recomputes step 4 for sessions already in runs/v2_validation.json without running anything.
    --max-evals N uses only the first N evaluations of each session (equal-budget comparisons); results are stored
    under "<session>@N". A config already run on the validation period is not run again: its ledger row is reused.
    --out writes to another file than runs/v2_validation.json.

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


def pooled_var(ledger) -> dict[str, float]:
    """Per template: variance of the screen-phase daily Sharpe over all P4 sessions."""
    rows = ledger.query("SELECT * FROM runs WHERE session LIKE 'p4-%' AND session NOT LIKE '%::%' "
                        "AND eval_split='screen' AND status='done'")
    srs: dict[str, list] = {}
    for r in rows:
        x = daily_ret(r["run_id"])
        if x is not None and len(x) > 20:
            srs.setdefault(json.loads(r["config_json"])["cfg"]["template"], []).append(stats.sharpe(x))
    return {t: float(np.nanvar(v, ddof=1)) for t, v in srs.items() if len(v) > 2}


def trial_stats(rows: list[dict], var: dict[str, float]) -> dict[str, dict]:
    """Per template: number of distinct configs tried in the session, pooled cross-trial variance."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["status"] == "failed" and (r["error"] or "").startswith("interrupted"):
            continue
        tpl = json.loads(r["config_json"])["cfg"].get("template")
        if tpl is None:
            continue
        out.setdefault(tpl, set()).add(r["config_hash"])
    return {t: {"n_trials": len(c), "var_sr_daily": var.get(t, float("nan"))} for t, c in out.items()}


def dsr_of(run_id: str, ts: dict):
    x = daily_ret(run_id)
    if x is None or not np.isfinite(ts["var_sr_daily"]):
        return None
    return stats.deflated_sharpe(x, ts["n_trials"], ts["var_sr_daily"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dsr-only", action="store_true")
    ap.add_argument("--max-evals", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    ledger = Ledger()
    p = Path(a.out) if a.out else runs_root() / "v2_validation.json"
    var = pooled_var(ledger)
    print("pooled cross-trial variance of daily Sharpe:", {k: round(v, 5) for k, v in var.items()}, flush=True)
    if a.dsr_only:
        val = json.loads(p.read_text())
        for s in a.sessions:
            ts_all = trial_stats(ledger.query("SELECT * FROM runs WHERE session=?", (s,)), var)
            val[s].pop("var_sr_daily", None)   # superseded by per-template values
            for tpl, t in val[s]["templates"].items():
                t.update(ts_all[tpl])
                for c in t["candidates"]:
                    c["dsr"] = dsr_of(c["valid_run"], ts_all[tpl])
            print(s, {k: (v["n_trials"], round(v["var_sr_daily"], 5)) for k, v in ts_all.items()}, flush=True)
        p.write_text(json.dumps(val, indent=1, default=str))
        print("saved", p)
        return
    for s in a.sessions:
        rows = ledger.query("SELECT * FROM runs WHERE session=? ORDER BY created_at, run_id", (s,))
        rows = [r for r in rows if not (r["status"] == "failed" and (r["error"] or "").startswith("interrupted"))]
        if a.max_evals:
            rows = rows[: a.max_evals]
        key = f"{s}@{a.max_evals}" if a.max_evals else s
        n_trials = len(rows)
        meta = json.loads((runs_root() / "sessions" / s / "session.json").read_text())
        settings = spec.Settings(**meta["settings"])
        folds = [r for r in rows if r["eval_split"] == "folds" and r["status"] == "done" and r["objective"] is not None]
        ts_all = trial_stats(rows, var)
        by_t: dict[str, list] = {}
        for r in folds:
            by_t.setdefault(json.loads(r["config_json"])["template"], []).append(r)
        sess_out = {"n_trials": n_trials, "arm": meta["arm"], "templates": {}}
        for tpl, rs in by_t.items():
            cands = sorted(rs, key=lambda r: -r["objective"])[: a.k]
            evals = []
            for r in cands:
                cfg = json.loads(r["config_json"])["cfg"]
                done = ledger.query("SELECT run_id, metrics_json FROM runs WHERE config_hash=? AND eval_split='valid' "
                                    "AND status='done' ORDER BY created_at LIMIT 1", (r["config_hash"],))
                if done:
                    res = {"run_id": done[0]["run_id"], "metrics": json.loads(done[0]["metrics_json"])}
                    print("reuse", res["run_id"], flush=True)
                else:
                    res = run_pipeline(cfg, settings, phase="valid", source="evaluator", session=f"{s}::valid",
                                       ledger=ledger, verbose=True)
                evals.append({"folds_run": r["run_id"], "folds_objective": r["objective"], "valid_run": res["run_id"],
                              "valid_objective": res["metrics"]["objective"], "valid_sharpe": res["metrics"]["sharpe"],
                              "valid_bp_per_day": res["metrics"]["mean_bp_per_day"],
                              "valid_x2_objective": res["metrics"]["x2_objective"],
                              "dsr": dsr_of(res["run_id"], ts_all[tpl]),
                              "config": cfg})
            best = max(evals, key=lambda e: e["valid_objective"] if np.isfinite(e["valid_objective"]) else -1e9)
            sess_out["templates"][tpl] = {"candidates": evals, "finalist": best["valid_run"], **ts_all[tpl]}
        print(json.dumps({t: {"finalist": v["finalist"]} for t, v in sess_out["templates"].items()}), flush=True)
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev[key] = sess_out
        p.write_text(json.dumps(prev, indent=1, default=str))
        print("saved", s, p, flush=True)   # saved after every session


if __name__ == "__main__":
    main()
