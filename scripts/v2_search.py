"""Run one v2 search session (agent or random arm). Resumable with --name.

    conda run -n minute-agent python scripts/v2_search.py random --budget 100 --seed 0 [--t0] [--sizing]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["agent", "random"])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--t0", action="store_true", help="allow_t0 switch")
    ap.add_argument("--sizing", action="store_true", help="position_sizing switch")
    ap.add_argument("--costs", default="default")
    ap.add_argument("--ledger")
    a = ap.parse_args()
    from minute_agent.agent.loop_v2 import SessionV2, agent_search, random_search
    from minute_agent.pipeline import spec
    st = spec.Settings(position_sizing=a.sizing, allow_t0=a.t0, costs=a.costs)
    sess = SessionV2(a.arm, a.budget, st, name=a.name, seed=a.seed,
                     ledger_path=Path(a.ledger) if a.ledger else None)
    print("session:", sess.name, flush=True)
    if a.arm == "random":
        random_search(sess)
    else:
        from minute_agent.agent.llm import LLMClient
        agent_search(sess, LLMClient(log_path=sess.dir / "llm_calls.jsonl", temperature=0.7 + 0.1 * (a.seed % 3)))


if __name__ == "__main__":
    main()
