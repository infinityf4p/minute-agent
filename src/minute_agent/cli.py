"""Command line entry point.

    python -m minute_agent.cli sim --n-stocks 500
    python -m minute_agent.cli run configs/experiments/baseline_lgbm.yaml
    python -m minute_agent.cli ledger
"""

from __future__ import annotations

import argparse
from datetime import date


def main(argv=None):
    ap = argparse.ArgumentParser(prog="minute-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="generate a simulated A-share dataset")
    s.add_argument("--dataset", default="sim_cn")
    s.add_argument("--n-stocks", type=int, default=500)
    s.add_argument("--start", default="2016-01-01")
    s.add_argument("--end", default="2026-10-31")
    s.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="run one experiment config")
    r.add_argument("config")
    r.add_argument("--source", default="human")

    lg = sub.add_parser("ledger", help="show recent ledger rows")
    lg.add_argument("-n", type=int, default=20)

    a = ap.parse_args(argv)
    if a.cmd == "sim":
        from minute_agent.data import simulate
        simulate.generate(a.dataset, simulate.SimConfig(
            n_stocks=a.n_stocks, start=date.fromisoformat(a.start), end=date.fromisoformat(a.end), seed=a.seed))
    elif a.cmd == "run":
        from minute_agent.experiment.runner import run
        run(a.config, source=a.source)
    elif a.cmd == "ledger":
        from minute_agent.experiment.ledger import Ledger
        for row in Ledger().rows(a.n):
            ic = f"{row['ic']:.4f}" if row["ic"] is not None else "  -   "
            ex = f"{row['excess_net_bp']:.1f}bp" if row["excess_net_bp"] is not None else "-"
            tt = f"{row['t_total']:.0f}s" if row["t_total"] is not None else "-"
            print(f"{row['run_id']}  {row['status']:7s} {row['target'] or '':9s} IC {ic}  net {ex}  {tt}")


if __name__ == "__main__":
    main()
