"""P5: open the locked test period ONCE and evaluate every finalist plus fixed baselines.

Inputs: runs/v2_validation.json (finalists chosen on validation by scripts/v2_validate.py).
Everything evaluated here is decided before the opening. Results go to runs/final/v2_test.json and the
opening is logged in runs/locked_test_openings.jsonl. A second run refuses to start.

    set MINUTE_AGENT_UNLOCK_TEST=<phrase>   (see minute_agent.split)
    conda run -n minute-agent python scripts/v2_final.py <session> [<session> ...]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minute_agent import split  # noqa: E402
from minute_agent.experiment.ledger import Ledger  # noqa: E402
from minute_agent.paths import runs_root  # noqa: E402
from minute_agent.pipeline import spec  # noqa: E402
from minute_agent.pipeline.run import run_pipeline  # noqa: E402
from minute_agent.report import stats  # noqa: E402

ST_SWITCH = date(2026, 7, 6)
FEATS = [{"op": "ret", "window_min": [5, 30, 60, 240]}, {"op": "realized_vol", "window_min": [30, 240]},
         {"op": "vwap_dev", "at": "decision_time"}, {"op": "volume_profile", "buckets": ["open30", "mid", "last30"]},
         {"op": "last_n_min_stats", "n": 30}, {"op": "jump_count", "threshold_sigma": 4},
         {"op": "daily_lag", "fields": ["ret", "turnover"], "lags": [1, 5, 20]}]
COMMON = {"features": FEATS, "label_norm": "rank", "cross_section_norm": "rank",
          "model": {"type": "xgb", "params": {"max_depth": 6, "lr": 0.05, "n_estimators": 200}},
          "train": {"window_years": 3, "retrain_every": "quarter"}, "execution": {"style": "next_vwap"},
          "universe_filter": {"min_price": 2, "min_adv_pct": 20}}
BASELINES = {
    "baseline_T3_overnight_top10": {**COMMON, "template": "T3", "decision_time": "14:50", "target": {"kind": "overnight"},
                                    "portfolio": {"top_frac": 0.1}},
    "baseline_T2_overnight_sell": {**COMMON, "template": "T2", "target": {"kind": "overnight"},
                                   "t0": {"mode": "overnight", "direction": "sell_first", "size_frac": 0.5, "entry_q": 0.9}},
    "baseline_T2_intraday": {**COMMON, "template": "T2", "t0_grid": "every_30", "target": {"kind": "to_flatten"},
                             "t0": {"mode": "intraday", "direction": "both", "size_frac": 0.33, "entry_q": 0.9, "max_trips": 1}},
}


def main():
    sessions = sys.argv[1:]
    out_path = runs_root() / "final" / "v2_test.json"
    if out_path.exists():
        raise SystemExit(f"the test period was already opened for v2 ({out_path}); it is opened only once")
    val = json.loads((runs_root() / "v2_validation.json").read_text())
    ledger = Ledger()
    entries = []
    for s in sessions:
        meta = json.loads((runs_root() / "sessions" / s / "session.json").read_text())
        for tpl, t in val[s]["templates"].items():
            fin = next(c for c in t["candidates"] if c["valid_run"] == t["finalist"])
            entries.append({"label": f"{s}:{tpl}", "session": s, "arm": meta["arm"], "settings": meta["settings"],
                            "config": fin["config"], "valid_objective": fin["valid_objective"],
                            "n_trials": t["n_trials"], "var_sr": t["var_sr_daily"]})   # per template
    full = spec.Settings(position_sizing=True, allow_t0=True).as_dict()
    for k, cfg in BASELINES.items():
        entries.append({"label": k, "arm": "baseline", "settings": full, "config": cfg})
    print(json.dumps([{k: v for k, v in e.items() if k != "config"} for e in entries], indent=1, default=str))

    results = {"entries": []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with split.final_test_unlock(f"v2 final: {len(entries)} pipelines"):
        for e in entries:
            try:
                r = run_pipeline(e["config"], spec.Settings(**e["settings"]), phase="test", source="final",
                                 session="v2-final-test", ledger=ledger, verbose=True)
                ser = pl.read_parquet(runs_root() / r["run_id"] / "daily.parquet")
                x = ser["ret"].to_numpy()
                t, p = stats.t_test_mean(x)
                pre = ser.filter(pl.col("date") < ST_SWITCH)["ret"].to_numpy()
                post = ser.filter(pl.col("date") >= ST_SWITCH)["ret"].to_numpy()
                row = {**{k: v for k, v in e.items() if k != "config"}, "status": "done", "run_id": r["run_id"],
                       "metrics": {k: v for k, v in r["metrics"].items() if not isinstance(v, dict)},
                       "t": t, "p": p,
                       "bp_per_day_before_2026_07_06": float(np.nanmean(pre) * 1e4),
                       "bp_per_day_after_2026_07_06": float(np.nanmean(post) * 1e4) if len(post) else None}
                if e.get("n_trials") and np.isfinite(e.get("var_sr", np.nan)):
                    row["dsr"] = stats.deflated_sharpe(x, e["n_trials"], e["var_sr"])
            except Exception as ex:
                row = {**{k: v for k, v in e.items() if k != "config"}, "status": "failed", "error": repr(ex)}
            results["entries"].append(row)
            out_path.write_text(json.dumps(results, indent=1, default=str))   # also blocks a second opening
    results["holm"] = stats.holm({r["label"]: r["p"] for r in results["entries"] if r["status"] == "done"})
    out_path.write_text(json.dumps(results, indent=1, default=str))
    print("saved", out_path)


if __name__ == "__main__":
    main()
