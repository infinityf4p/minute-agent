"""Child-process entry point used by agent.executor."""

from __future__ import annotations

import argparse
import json
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--rationale-file")
    a = ap.parse_args()

    from minute_agent.agent.executor import summarize
    from minute_agent.experiment.ledger import Ledger
    from minute_agent.experiment.runner import run

    cfg = json.load(open(a.config))
    rationale = open(a.rationale_file, encoding="utf-8").read() if a.rationale_file else None
    try:
        r = run(cfg, source=a.source, ledger=Ledger(a.ledger), verbose=False, save=True,
                run_id=a.run_id, session=a.session, rationale=rationale)
        t = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r["timings"].items()}
        out = {"status": "done", "metrics": summarize(r["metrics"]), "timings": t}
    except Exception as e:
        out = {"status": "failed", "error": "".join(traceback.format_exception_only(e))[-1500:],
               "trace": traceback.format_exc()[-3000:]}
    with open(a.out, "w") as f:
        json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
