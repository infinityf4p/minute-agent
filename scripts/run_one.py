"""Run one v2 config file through the pipeline (for debugging and spot checks).

    conda run -n minute-agent python scripts/run_one.py <config.json> <phase> [--t0] [--sizing] [--ledger path]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("phase")
    ap.add_argument("--t0", action="store_true")
    ap.add_argument("--sizing", action="store_true")
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--ledger", default="runs/dev_ledger.sqlite")
    a = ap.parse_args()
    from minute_agent.experiment.ledger import Ledger
    from minute_agent.pipeline import spec
    from minute_agent.pipeline.run import run_pipeline
    r = run_pipeline(json.load(open(a.config)), spec.Settings(position_sizing=a.sizing, allow_t0=a.t0, custom_features=a.custom), phase=a.phase,
                     ledger=Ledger(a.ledger), verbose=True, save=False)
    print(json.dumps({k: v for k, v in r["metrics"].items() if not isinstance(v, dict)}, default=str)[:1500])
    print("timings", r["metrics"]["timings"], flush=True)


if __name__ == "__main__":
    main()
