"""Time the M1 pipeline: cold feature build, then train + evaluate with a warm cache.

    conda run -n minute-agent python scripts/benchmark.py --dataset sim_cn
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="sim_cn")
ap.add_argument("--config", default="configs/experiments/baseline_lgbm.yaml")
ap.add_argument("--start", default="2018-12-01", help="feature build start (3y window before 2022)")
ap.add_argument("--end", default="2023-12-31")
ap.add_argument("--out", default=None)
args = ap.parse_args()

REPO = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.environ["MINUTE_AGENT_CACHE"] = str(REPO / "cache" / f"bench-{args.dataset}-{stamp}")
sys.path.insert(0, str(REPO / "src"))

from minute_agent.experiment import config as C  # noqa: E402
from minute_agent.experiment.ledger import Ledger  # noqa: E402
from minute_agent.experiment.runner import run  # noqa: E402
from minute_agent.features.cache import FeatureCache  # noqa: E402

base = C.load(REPO / args.config)
base["dataset"] = args.dataset
result = {"dataset": args.dataset, "cache": os.environ["MINUTE_AGENT_CACHE"]}

# 1) cold feature build
fc = FeatureCache(args.dataset)
t0 = time.time()
st = fc.build(base["features"], date.fromisoformat(args.start), date.fromisoformat(args.end))
cold = time.time() - t0
result["features_cold"] = {
    "seconds": round(cold, 1), "minute_rows": st.minute_rows, "months": st.months_built,
    "rows_per_s": round(st.minute_rows / max(st.seconds_read + st.seconds_compute, 1e-9)),
    "read_s": round(st.seconds_read, 1), "compute_s": round(st.seconds_compute, 1),
    "leak_check_s": round(st.seconds_leak_check, 1),
    "per_op_s": {k: round(v, 1) for k, v in st.per_op_seconds.items()},
}
print("features cold:", json.dumps(result["features_cold"]), flush=True)

# 2) warm runs: one training+evaluation round per variant
variants = {
    "lgbm_500_quarterly": {},
    "lgbm_200_quarterly": {"model": {"type": "lgbm", "params": {"num_leaves": 63, "lr": 0.1, "n_estimators": 200}}},
    "lgbm_500_yearly": {"train": {"window_years": 3, "retrain_every": "year"}},
    "linear_quarterly": {"model": {"type": "linear", "params": {"alpha": 10.0}}},
}
ledger = Ledger(REPO / "runs" / "bench_ledger.sqlite")
result["rounds"] = {}
for name, patch in variants.items():
    cfg = copy.deepcopy(base)
    cfg.update(patch)
    cfg["name"] = name
    r = run(cfg, source="benchmark", ledger=ledger, verbose=False)
    m, t = r["metrics"], r["timings"]
    result["rounds"][name] = {
        "total_s": round(t["total"], 1), "features_s": round(t["features"], 1), "panel_s": round(t["panel"], 1),
        "train_s": round(t["train"], 1), "eval_s": round(t["eval"], 2), "n_models": t["n_models"],
        "avg_train_rows": t["avg_train_rows"], "ic": round(m["ic"], 4), "rank_ic": round(m["rank_ic"], 4),
        "icir": round(m["icir"], 3), "excess_gross_bp": round(m["excess_gross_bp"], 1),
        "excess_net_bp": round(m["excess_net_bp"], 1),
    }
    print(name, json.dumps(result["rounds"][name]), flush=True)

out = Path(args.out) if args.out else REPO / "runs" / f"benchmark-{args.dataset}-{stamp}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print("saved", out)
