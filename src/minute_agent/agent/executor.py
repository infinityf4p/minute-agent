"""Run one experiment config in a separate process with a time limit.

The child process gets a copy of the environment WITHOUT the locked-test
unlock variable, so nothing an experiment does can open the test period.
Docker isolation (PLAN section 3) is the next step once WSL2 is enabled; the
subprocess boundary + static checks on custom code + the read guard are the
current sandbox.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from minute_agent import split
from minute_agent.experiment import config as C
from minute_agent.experiment.ledger import Ledger
from minute_agent.experiment.runner import new_run_id

SUMMARY_KEYS = [
    "ic", "rank_ic", "icir", "rank_icir", "ic_pos_frac", "group_ret_bp", "top_minus_bottom_bp", "monotonicity",
    "long_gross_bp", "universe_bp", "excess_gross_bp", "cost_bp", "excess_net_bp", "excess_net_ann",
    "excess_net_sharpe", "excess_net_maxdd", "top_turnover", "avg_top_names", "avg_universe_names",
    "n_features", "importance_top", "objective",
]


def summarize(metrics: dict) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        v = metrics.get(k)
        if isinstance(v, float):
            v = round(v, 5)
        elif isinstance(v, list):
            v = [round(x, 2) if isinstance(x, float) else x for x in v]
        elif isinstance(v, dict):
            v = {kk: round(vv, 4) for kk, vv in list(v.items())[:10]}
        out[k] = v
    return out


def execute(cfg: dict, source: str, session: str, rationale: str | None, ledger_path: Path,
            workdir: Path, timeout_s: int = 1800) -> dict:
    cfg = C.validate(cfg)
    run_id = new_run_id(C.config_hash(cfg))
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_path = workdir / f"{run_id}.config.json"
    out_path = workdir / f"{run_id}.result.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    env = {k: v for k, v in os.environ.items() if k != split.UNLOCK_ENV}
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "minute_agent.agent.execute", str(cfg_path), str(out_path),
           "--run-id", run_id, "--source", source, "--session", session, "--ledger", str(ledger_path)]
    if rationale:
        (workdir / f"{run_id}.rationale.txt").write_text(rationale, encoding="utf-8")
        cmd += ["--rationale-file", str(workdir / f"{run_id}.rationale.txt")]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout_s, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        err = proc.stderr[-3000:] if proc.returncode else None
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout_s}s"
    secs = time.time() - t0
    if out_path.exists():
        res = json.loads(out_path.read_text())
        if res.get("status") == "done":
            return {"run_id": run_id, "status": "done", "seconds": round(secs, 1), **res}
        err = res.get("error") or err
    try:   # a crash or timeout leaves the ledger row 'running'
        Ledger(ledger_path).fail(run_id, err or "unknown failure", {"total": secs})
    except Exception:
        pass
    return {"run_id": run_id, "status": "failed", "seconds": round(secs, 1), "error": (err or "")[-1500:]}
