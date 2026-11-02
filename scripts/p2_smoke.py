"""P2 smoke test: run hand-written configs for each template through the v2 pipeline.

    conda run -n minute-agent python scripts/p2_smoke.py <phase> T3 T1 T2 T2on
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
import sys, json
from minute_agent.pipeline import spec
from minute_agent.pipeline.run import run_pipeline
from minute_agent.experiment.ledger import Ledger
from minute_agent.paths import runs_root
FEATS = [{"op": "ret", "window_min": [5, 30, 60, 240]}, {"op": "realized_vol", "window_min": [30, 240]},
         {"op": "vwap_dev", "at": "decision_time"}, {"op": "volume_profile", "buckets": ["open30", "mid", "last30"]},
         {"op": "last_n_min_stats", "n": 30}, {"op": "jump_count", "threshold_sigma": 4},
         {"op": "daily_lag", "fields": ["ret", "turnover"], "lags": [1, 5, 20]}]
COMMON = {"features": FEATS, "label_norm": "rank", "cross_section_norm": "rank",
          "model": {"type": "xgb", "params": {"max_depth": 6, "lr": 0.05, "n_estimators": 200}},
          "train": {"window_years": 3, "retrain_every": "quarter"}, "execution": {"style": "next_vwap"},
          "universe_filter": {"min_price": 2, "min_adv_pct": 20}}
CFGS = {
 "T3": {**COMMON, "template": "T3", "decision_time": "14:50", "target": {"kind": "overnight"},
        "portfolio": {"top_frac": 0.1}},
 "T1": {**COMMON, "template": "T1", "decision_time": "14:30", "target": {"kind": "fwd_days", "days": 5},
        "portfolio": {"top_frac": 0.05, "buffer_frac": 1.0, "rebalance_every": 1, "hold_min": 2}},
 "T2": {**COMMON, "template": "T2", "t0_grid": "every_30", "target": {"kind": "to_flatten"},
        "t0": {"mode": "intraday", "direction": "both", "size_frac": 0.33, "entry_q": 0.9, "max_trips": 1}},
 "T2on": {**COMMON, "template": "T2", "target": {"kind": "overnight"},
        "t0": {"mode": "overnight", "direction": "sell_first", "size_frac": 0.5, "entry_q": 0.9}},
}
if __name__ == "__main__":
    s = spec.Settings(position_sizing=False, allow_t0=True)
    L = Ledger(runs_root() / "dev_ledger.sqlite")
    for k in sys.argv[2:]:
        r = run_pipeline(CFGS[k], s, phase=sys.argv[1], ledger=L, verbose=True, save=False)
        m = r["metrics"]
        print(k, json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in m.items() if x not in ("timings",)}, default=str))
        print("  timings", m["timings"], flush=True)
