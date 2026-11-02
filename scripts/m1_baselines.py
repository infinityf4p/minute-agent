"""M1 acceptance: run the fixed baselines on the real dataset (validation 2022-2023)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    from minute_agent.evaluator import headline
    from minute_agent.experiment import config as C
    from minute_agent.experiment.ledger import Ledger
    from minute_agent.experiment.runner import run

    ledger = Ledger()
    for r in ledger.query("SELECT run_id FROM runs WHERE session='m1-baselines' AND status='running'"):
        ledger.fail(r["run_id"], "interrupted")
    done = {r["config_hash"] for r in ledger.query("SELECT config_hash FROM runs WHERE session='m1-baselines' AND status='done'")}

    names = sys.argv[1:] or [f"{m}_{t}" for t in ("overnight", "next_cc")
                             for m in ("baseline_lgbm", "baseline_xgb", "gru", "daily_only_lgbm")]
    for n in names:
        path = REPO / "configs" / "experiments" / "cn" / f"{n}.yaml"
        if C.config_hash(C.load(path)) in done:
            print(f"{n}: already done, skipped", flush=True)
            continue
        r = run(REPO / "configs" / "experiments" / "cn" / f"{n}.yaml", source="baseline", verbose=False,
                session="m1-baselines")
        t = r["timings"]
        print(f"{n:28s} {headline(r['metrics'])} | obj {r['metrics']['objective']:.2f} | "
              f"features {t['features']:.0f}s train {t['train']:.0f}s total {t['total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
