"""Run P4 search sessions one after another in a lane (so at most one session per lane uses memory).

Waits until the session given by --after has used its budget, then runs the listed seeds in order.
Sessions are resumable, so re-launching a lane continues where it stopped.

    conda run -n minute-agent python scripts/p4_lane.py agent --seeds 1 2 --after p4-agent-s0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["agent", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--after")
    ap.add_argument("--budget", type=int, default=100)
    a = ap.parse_args()
    from minute_agent.agent.loop_v2 import SessionV2, agent_search, random_search
    from minute_agent.experiment.ledger import Ledger
    from minute_agent.pipeline import spec

    st = spec.Settings(position_sizing=True, allow_t0=True)
    if a.after:
        led = Ledger()
        while True:
            n = led.query("SELECT COUNT(*) AS n FROM runs WHERE session=? AND NOT (status='failed' AND error LIKE "
                          "'interrupted%') AND status != 'running'", (a.after,))[0]["n"]
            if n >= a.budget:
                break
            print(f"waiting for {a.after}: {n}/{a.budget}", flush=True)
            time.sleep(300)
        time.sleep(60)   # let the finishing process exit
    for seed in a.seeds:
        sess = SessionV2(a.arm, a.budget, st, name=f"p4-{a.arm}-s{seed}", seed=seed)
        print("session:", sess.name, flush=True)
        if a.arm == "random":
            random_search(sess)
        else:
            from minute_agent.agent.llm import LLMClient
            agent_search(sess, LLMClient(log_path=sess.dir / "llm_calls.jsonl", temperature=0.7 + 0.1 * (seed % 3)))


if __name__ == "__main__":
    main()
