"""Child-process entry for one v2 evaluation (used by agent.loop_v2 through a subprocess)."""

from __future__ import annotations

import argparse
import json
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--rationale-file")
    a = ap.parse_args()

    from minute_agent.experiment.ledger import Ledger
    from minute_agent.pipeline import spec
    from minute_agent.pipeline.run import run_pipeline

    cfg = json.load(open(a.config))
    st = spec.Settings(**json.load(open(a.settings)))
    rationale = open(a.rationale_file, encoding="utf-8").read() if a.rationale_file else None
    try:
        r = run_pipeline(cfg, st, phase=a.phase, source=a.source, session=a.session, rationale=rationale,
                         ledger=Ledger(a.ledger), run_id=a.run_id, verbose=False, save=True)
        keep = ("objective", "sharpe", "mean_bp_per_day", "ann_ret", "maxdd", "hit", "cost_bp_per_day", "turnover",
                "x2_objective", "gross_objective", "rank_ic", "fold1_objective", "fold2_objective", "avg_positions",
                "avg_exposure", "t0_round_trips_per_stock_day", "t0_win_rate", "t0_carried_overnight_frac",
                "book_rank_ic", "t0_rank_ic")
        m = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r["metrics"].items() if k in keep}
        out = {"status": "done", "metrics": m, "timings": {k: round(v, 1) for k, v in r["timings"].items()}}
    except Exception as e:
        out = {"status": "failed", "error": "".join(traceback.format_exception_only(e))[-1500:],
               "trace": traceback.format_exc()[-3000:]}
    with open(a.out, "w") as f:
        json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
