"""M5: pick pipelines on validation, open the locked test set once, correct for trials, write the report data.

    MINUTE_AGENT_UNLOCK_TEST=<phrase> conda run -n minute-agent python scripts/m5_final.py \
        --agent <session> [--agent <session> ...] --random <session> [--k 5]

Selection uses validation results only. Everything that will be looked at on
the test set is decided before the single opening:
  * top-K per search session by the validation objective
  * fixed baselines (PLAN 5.1, 5.2, 5.4)
  * ablations of the best agent pipeline: daily features only / minute features only
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minute_agent.experiment import config as C  # noqa: E402
from minute_agent.experiment.ledger import Ledger  # noqa: E402
from minute_agent.paths import runs_root  # noqa: E402
from minute_agent.report import stats  # noqa: E402
from minute_agent.report.final import final_evaluation  # noqa: E402

DAILY_OPS = {"daily_lag", "daily_stats"}
BASELINES = ["baseline_lgbm_overnight", "baseline_lgbm_next_cc", "baseline_xgb_overnight", "baseline_xgb_next_cc",
             "gru_overnight", "gru_next_cc", "daily_only_lgbm_overnight", "daily_only_lgbm_next_cc"]


def session_rows(ledger: Ledger, session: str) -> list[dict]:
    return ledger.query("SELECT * FROM runs WHERE session=? AND status='done' AND objective IS NOT NULL", (session,))


def valid_daily_excess(run_id: str) -> np.ndarray | None:
    p = runs_root() / run_id / "portfolio_daily.parquet"
    return pl.read_parquet(p)["excess_net"].to_numpy() if p.exists() else None


def ablations(cfg: dict) -> dict[str, dict]:
    out = {}
    daily = [f for f in cfg["features"] if f["op"] in DAILY_OPS]
    minute = [f for f in cfg["features"] if f["op"] not in DAILY_OPS]
    std_daily = [{"op": "daily_lag", "fields": ["ret", "turnover"], "lags": [1, 2, 3, 5, 10, 20, 60]},
                 {"op": "daily_stats", "windows": [5, 10, 20, 60, 120]}]
    if cfg["model"]["type"] not in ("gru_seq", "transformer_seq"):
        a = copy.deepcopy(cfg)
        a["features"] = daily or std_daily
        out["daily_only"] = a
    if minute and daily:
        b = copy.deepcopy(cfg)
        b["features"] = minute
        out["minute_only"] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", default=[])
    ap.add_argument("--random", action="append", default=[])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="only print the selection")
    a = ap.parse_args()

    ledger = Ledger()
    entries, selection = [], {}
    for kind, sessions in (("agent", a.agent), ("random", a.random)):
        for s in sessions:
            rows = sorted(session_rows(ledger, s), key=lambda r: -r["objective"])
            selection[s] = {"n_done": len(rows), "n_total": len(ledger.query("SELECT run_id FROM runs WHERE session=?", (s,)))}
            for i, r in enumerate(rows[: a.k]):
                cfg = json.loads(r["config_json"])
                entries.append({"label": f"{s}#{i + 1}", "group": kind, "session": s, "valid_run_id": r["run_id"],
                                "valid_objective": r["objective"], "config": cfg})
    for b in BASELINES:
        cfg = C.load(REPO / "configs" / "experiments" / "cn" / f"{b}.yaml")
        entries.append({"label": b, "group": "baseline", "config": cfg})
    best_agent = next((e for e in entries if e["group"] == "agent"), None)
    if best_agent:
        for name, cfg in ablations(best_agent["config"]).items():
            entries.append({"label": f"ablation_{name}", "group": "ablation", "config": cfg})

    print(json.dumps([{k: v for k, v in e.items() if k != "config"} for e in entries], indent=1, default=str))
    if a.dry_run:
        return

    res = final_evaluation(entries, reason="M5 final evaluation")

    # --- statistics
    report = {"selection": selection, "entries": []}
    for e in res["entries"]:
        row = {k: v for k, v in e.items() if k not in ("config",)}
        port = runs_root() / "final" / f"{e['label']}.portfolio.parquet"
        if e["status"] == "done" and port.exists():
            x = pl.read_parquet(port)["excess_net"].to_numpy()
            row["test_t"], row["test_p"] = stats.t_test_mean(x)
        report["entries"].append(row)
    pvals = {r["label"]: r["test_p"] for r in report["entries"] if "test_p" in r}
    report["holm"] = stats.holm(pvals)

    # deflated Sharpe of each session's validation winner, using all of that session's trials
    report["deflated_sharpe_valid"] = {}
    for s in a.agent + a.random:
        rows = session_rows(ledger, s)
        srs = [stats.sharpe(x) for r in rows if (x := valid_daily_excess(r["run_id"])) is not None]
        srs = [v for v in srs if np.isfinite(v)]
        if len(srs) < 3:
            continue
        best = max(rows, key=lambda r: r["objective"])
        x = valid_daily_excess(best["run_id"])
        n_trials = selection[s]["n_total"]
        report["deflated_sharpe_valid"][s] = stats.deflated_sharpe(x, n_trials, float(np.var(srs, ddof=1)))
    out = runs_root() / "final" / "m5_report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
