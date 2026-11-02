"""M5: the one-time evaluation on the locked test period (2024-01 .. 2026-10).

All pipelines to be tested are passed in ONE call. The function refuses to run
if runs/FINAL_TEST_RESULTS.json already exists, so the test set is opened once.
Each pipeline is retrained on a rolling basis exactly as on validation, with
training windows that end before each test retrain date (so they may use
2016-2023 data, including the validation years).
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

import polars as pl

from minute_agent import split
from minute_agent.data import store
from minute_agent.evaluator import evaluate
from minute_agent.evaluator.integrity import verify as verify_evaluator
from minute_agent.experiment import config as C
from minute_agent.experiment.runner import build_panel, rolling_predict
from minute_agent.market import load_market
from minute_agent.paths import runs_root

RESULTS = "FINAL_TEST_RESULTS.json"


def _evaluate_on_test(cfg: dict, verbose: bool) -> tuple[dict, pl.DataFrame]:
    cfg = C.validate(cfg)
    ev = split.LOCKED_TEST
    start = max(split.TRAIN.start, ev.start - timedelta(days=int(365.25 * float(cfg["train"]["window_years"])) + 10))
    timings: dict = {}
    df, feat_cols = build_panel(cfg, start, ev.end, timings, verbose, feature_workers=1)
    cal = store.read_calendar(cfg["dataset"], start, ev.end)
    pred, _ = rolling_predict(cfg, df, feat_cols, cal, ev.start, ev.end, timings, verbose)
    metrics = evaluate(pred, load_market(cfg["market"]), cfg["target"], allow_locked=True)
    port = metrics.pop("series")["portfolio"]
    metrics["timings"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in timings.items()}
    return metrics, port


def final_evaluation(entries: list[dict], reason: str, verbose: bool = True) -> dict:
    """entries: [{"label", "group", "config", "valid_run_id"(optional)}]"""
    out_dir = runs_root() / "final"
    marker = runs_root() / RESULTS
    if marker.exists():
        raise RuntimeError(f"the locked test set was already opened (see {marker}); it is opened only once")
    labels = [e["label"] for e in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels")
    ev_hash = verify_evaluator()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"reason": reason, "evaluator_hash": ev_hash, "entries": []}
    with split.final_test_unlock(reason + f" | {len(entries)} pipelines: {labels}"):
        for e in entries:
            t0 = time.time()
            if verbose:
                print(f"[final] {e['label']} ...", flush=True)
            try:
                metrics, port = _evaluate_on_test(e["config"], verbose=False)
                port.write_parquet(out_dir / f"{e['label']}.portfolio.parquet")
                results["entries"].append({**e, "status": "done", "metrics": metrics,
                                           "seconds": round(time.time() - t0, 1)})
                if verbose:
                    print(f"   IC {metrics['ic']:.4f} RankIC {metrics['rank_ic']:.4f} net {metrics['excess_net_bp']:.1f}bp "
                          f"SR {metrics['excess_net_sharpe']:.2f} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as ex:
                results["entries"].append({**e, "status": "failed", "error": repr(ex)})
                if verbose:
                    print("   failed:", repr(ex), flush=True)
            # write after every pipeline so a crash keeps what was done (and still blocks a re-run)
            marker.write_text(json.dumps(results, indent=1, default=str))
    return results
