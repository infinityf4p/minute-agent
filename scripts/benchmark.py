"""Time the M1 pipeline: cold feature build, then one train + evaluate round per model.

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

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim_cn")
    ap.add_argument("--config", default="configs/experiments/baseline_lgbm.yaml")
    ap.add_argument("--start", default="2018-12-01", help="feature build start (3y window before 2022)")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--feature-workers", type=int, default=3)
    ap.add_argument("--seq-months", type=int, default=3,
                    help="also build this many months single-process, to compare speed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_cache = REPO / "cache" / f"bench-{args.dataset}-{stamp}"
    os.environ["MINUTE_AGENT_CACHE"] = str(bench_cache)
    sys.path.insert(0, str(REPO / "src"))

    from minute_agent.experiment import config as C
    from minute_agent.experiment.ledger import Ledger
    from minute_agent.experiment.runner import run
    from minute_agent.features.cache import FeatureCache

    base = C.load(REPO / args.config)
    base["dataset"] = args.dataset
    result = {"dataset": args.dataset, "cache": str(bench_cache), "cpu_count": os.cpu_count()}
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    # 0) single-process reference on a few months (separate cache dir)
    if args.seq_months:
        os.environ["MINUTE_AGENT_CACHE"] = str(bench_cache) + "-seq"
        y, m = start.year, start.month + args.seq_months
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        seq_end = date(y, m, 1)
        st = FeatureCache(args.dataset).build(base["features"], start, seq_end, workers=1)
        result["features_single_process"] = {"months": st.months_built, "minute_rows": st.minute_rows,
                                             "wall_s": round(st.seconds_wall, 1),
                                             "s_per_month": round(st.seconds_wall / max(st.months_built, 1), 2)}
        print("features single-process:", json.dumps(result["features_single_process"]), flush=True)
        os.environ["MINUTE_AGENT_CACHE"] = str(bench_cache)

    # 1) cold feature build, multi-process
    t0 = time.time()
    st = FeatureCache(args.dataset).build(base["features"], start, end, workers=args.feature_workers)
    result["features_cold"] = {
        "workers": args.feature_workers, "seconds": round(time.time() - t0, 1),
        "wall_s": round(st.seconds_wall, 1), "months": st.months_built, "minute_rows": st.minute_rows,
        "s_per_month": round(st.seconds_wall / max(st.months_built, 1), 2),
        "rows_per_s": round(st.minute_rows / max(st.seconds_wall, 1e-9)),
        "leak_check_s": round(st.seconds_leak_check, 1),
        "per_op_cpu_s": {k: round(v, 1) for k, v in st.per_op_seconds.items()},
    }
    print("features cold:", json.dumps(result["features_cold"]), flush=True)

    # 2) warm cache: one training + evaluation round per variant
    variants = {
        "lgbm_cpu_1worker": ({}, 1),
        "lgbm_cpu_4workers": ({}, 4),
        "xgb_gpu": ({"model": {"type": "xgb", "params": {"max_depth": 6, "lr": 0.05, "n_estimators": 500,
                                                          "device": "cuda"}}}, 1),
        "mlp_gpu": ({"model": {"type": "mlp", "params": {"epochs": 8, "device": "cuda"}}}, 1),
        "linear_cpu": ({"model": {"type": "linear", "params": {"alpha": 10.0}}}, 1),
    }
    ledger = Ledger(REPO / "runs" / "bench_ledger.sqlite")
    result["rounds"] = {}
    for name, (patch, tw) in variants.items():
        cfg = copy.deepcopy(base)
        cfg.update(patch)
        cfg["name"] = name
        r = run(cfg, source="benchmark", ledger=ledger, verbose=False, train_workers=tw)
        mt, t = r["metrics"], r["timings"]
        result["rounds"][name] = {
            "total_s": round(t["total"], 1), "load_features_s": round(t["features"], 1),
            "panel_s": round(t["panel"], 1), "train_s": round(t["train"], 1), "eval_s": round(t["eval"], 2),
            "device": t["train_device"], "train_workers": t["train_workers"], "n_models": t["n_models"],
            "avg_train_rows": t["avg_train_rows"], "ic": round(mt["ic"], 4), "rank_ic": round(mt["rank_ic"], 4),
            "icir": round(mt["icir"], 3), "excess_gross_bp": round(mt["excess_gross_bp"], 1),
            "excess_net_bp": round(mt["excess_net_bp"], 1),
        }
        print(name, json.dumps(result["rounds"][name]), flush=True)

    out = Path(args.out) if args.out else REPO / "runs" / f"benchmark-{args.dataset}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
