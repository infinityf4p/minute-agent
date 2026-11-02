"""Run one experiment config: features -> rolling training -> fixed evaluator -> ledger.

Rolling scheme: models are retrained at the start of every period
(train.retrain_every) inside the evaluation split. Each model is trained on
the window_years before the retrain date, minus an embargo of 1 trading day,
because the label of the last day before the retrain date is not yet known
when the model is trained the night before.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from minute_agent import panel as P
from minute_agent import split
from minute_agent.data import store
from minute_agent.evaluator import evaluate, headline
from minute_agent.evaluator.integrity import verify as verify_evaluator
from minute_agent.experiment import config as C
from minute_agent.experiment.ledger import Ledger
from minute_agent.features.cache import FeatureCache
from minute_agent.market import load_market
from minute_agent.features.extra_ops import seq_columns
from minute_agent.models import SEQ_MODEL_TYPES, cpu_count, make_model
from minute_agent.paths import REPO_ROOT, runs_root

BASE_COLS = {"price_dec", "amt_dec", "last_bidx"}
DEFAULT_TRAIN_WORKERS = 4      # parallel windows for CPU models
DEFAULT_FEATURE_WORKERS = int(os.environ.get("MINUTE_AGENT_FEATURE_WORKERS", 3))  # ~4GB RAM per worker at full-market size
EMBARGO_DAYS = 1
PERIOD_MONTHS = {"month": 1, "quarter": 3, "half": 6, "year": 12}


OBJECTIVE = "excess_net_sharpe"   # validation objective used to rank pipelines


def objective(metrics: dict) -> float:
    v = metrics.get(OBJECTIVE)
    return float(v) if v is not None and np.isfinite(v) else float("-inf")


def new_run_id(chash: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S-%f}-{chash}"


def retrain_dates(start: date, end: date, every: str) -> list[date]:
    if every == "never":
        return [start]
    step = PERIOD_MONTHS[every]
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        out.append(max(date(y, m, 1), start))
        m += step
        while m > 12:
            y, m = y + 1, m - 12
    return out


def _git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except OSError:
        return {"git_commit": None, "git_dirty": None}


def feature_specs(cfg: dict) -> list[dict]:
    specs = list(cfg["features"])
    if cfg["model"]["type"] in SEQ_MODEL_TYPES and not any(f["op"] == "seq_bars" for f in specs):
        specs.append({"op": "seq_bars", "bucket_min": 10})
    return specs


def build_panel(cfg: dict, start: date, end: date, timings: dict, verbose: bool,
                feature_workers: int = DEFAULT_FEATURE_WORKERS) -> tuple[pl.DataFrame, list[str]]:
    rules = load_market(cfg["market"])
    fc = FeatureCache(cfg["dataset"], cfg["market"], cfg.get("decision_time"))

    specs = feature_specs(cfg)
    t0 = time.time()
    st = fc.build(specs, start, end, verbose=verbose, workers=feature_workers)
    X = fc.load(specs, start, end)
    timings["features"] = time.time() - t0
    timings["features_detail"] = {"months_built": st.months_built, "months_cached": st.months_cached,
                                  "minute_rows": st.minute_rows, "read_s": round(st.seconds_read, 2),
                                  "compute_s": round(st.seconds_compute, 2),
                                  "leak_check_s": round(st.seconds_leak_check, 2), "wall_s": round(st.seconds_wall, 2)}

    t0 = time.time()
    cal = store.read_calendar(cfg["dataset"], start - timedelta(days=400), end)
    daily = store.read_daily(cfg["dataset"], start - timedelta(days=60), end)
    sec = store.read_security(cfg["dataset"])
    uni = P.universe(daily, sec, cal, X, rules.universe, cfg.get("universe_filter"))
    lab = P.labels(daily, cal, cfg["target"])
    feat_cols = [c for c in X.columns if c not in BASE_COLS and c not in ("symbol", "date")]
    df = (
        X.select(["symbol", "date", *feat_cols])
        .join(uni, on=["symbol", "date"], how="inner")
        .filter(pl.col("in_universe"))
        .join(lab, on=["symbol", "date"], how="left")
    )
    tab_cols = [c for c in feat_cols if c not in set(seq_columns(feat_cols))]
    df = P.cross_section_norm(df, tab_cols, cfg["cross_section_norm"])
    if cfg["label_norm"] == "rank":
        y = (pl.col("label").rank().over("date") - 1) / (pl.col("label").count().over("date") - 1).clip(lower_bound=1) - 0.5
    elif cfg["label_norm"] == "raw":
        y = pl.col("label")
    else:
        raise C.ConfigError(f"unknown label_norm {cfg['label_norm']!r}")
    df = df.with_columns(y.alias("y")).sort("date", "symbol")
    timings["panel"] = time.time() - t0
    return df, feat_cols


def _fit_window(cfg, df, feat_cols, p, nxt, train_lo, train_end_excl, threads):
    tr = df.filter(pl.col("date").is_between(train_lo, train_end_excl, closed="left") & pl.col("y").is_not_null())
    te = df.filter(pl.col("date").is_between(p, nxt, closed="left"))
    if tr.height == 0 or te.height == 0:
        return None
    model = make_model(cfg["model"], seed=int(cfg["seed"]))
    model.set_threads(threads)
    model.fit(tr.select(feat_cols).to_numpy(), tr["y"].to_numpy(), feat_cols)
    score = model.predict(te.select(feat_cols).to_numpy())
    pred = te.select("date", "symbol", "label", "buyable").with_columns(pl.Series("score", score))
    return p, tr.height, te.height, pred, model.importance(), model.device


def rolling_predict(cfg: dict, df: pl.DataFrame, feat_cols: list[str], cal: list[date],
                    ev_start: date, ev_end: date, timings: dict, verbose: bool,
                    workers: int | None = None) -> tuple[pl.DataFrame, dict]:
    """CPU models: windows train in parallel threads (LightGBM/XGBoost release the GIL,
    so this needs no data copies), cores split evenly. GPU models: one window at a time."""
    t0 = time.time()
    points = retrain_dates(ev_start, ev_end, cfg["train"]["retrain_every"])
    bounds = points[1:] + [ev_end + timedelta(days=1)]
    window = timedelta(days=int(round(365.25 * float(cfg["train"]["window_years"]))))
    cal_arr = np.array(cal, dtype="datetime64[D]")
    jobs = []
    for p, nxt in zip(points, bounds):
        k = int(np.searchsorted(cal_arr, np.datetime64(p)))
        jobs.append((p, nxt, p - window, cal[max(0, k - EMBARGO_DAYS)]))

    device = make_model(cfg["model"]).device
    ncpu = cpu_count()
    if device == "cpu" and len(jobs) > 1:
        w = max(1, min(len(jobs), workers or DEFAULT_TRAIN_WORKERS))
    else:
        w = 1
    threads = max(1, ncpu // w)
    if w == 1:
        results = [_fit_window(cfg, df, feat_cols, *j, threads) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=w) as ex:
            results = list(ex.map(lambda j: _fit_window(cfg, df, feat_cols, *j, threads), jobs))

    preds, importance, n_train = [], {}, []
    for r in results:
        if r is None:
            continue
        p, ntr, nte, pred, imp, dev = r
        preds.append(pred)
        n_train.append(ntr)
        for f, v in imp.items():
            importance[f] = importance.get(f, 0.0) + v
        if verbose:
            print(f"  retrain {p}: train {ntr:,} rows, predict {nte:,} rows ({dev})", flush=True)
    timings["train"] = time.time() - t0
    timings["train_device"] = device
    timings["train_workers"] = w
    timings["n_models"] = len(n_train)
    timings["avg_train_rows"] = int(np.mean(n_train)) if n_train else 0
    tot = sum(importance.values()) or 1.0
    importance = {k: v / tot for k, v in sorted(importance.items(), key=lambda kv: -kv[1])}
    return pl.concat(preds) if preds else pl.DataFrame(), importance


def run(cfg_in, source: str = "human", ledger: Ledger | None = None, verbose: bool = True,
        save: bool = True, feature_workers: int = DEFAULT_FEATURE_WORKERS,
        train_workers: int = DEFAULT_TRAIN_WORKERS, run_id: str | None = None,
        session: str | None = None, rationale: str | None = None) -> dict:
    t_all = time.time()
    cfg = C.load(cfg_in)
    chash = C.config_hash(cfg)
    run_id = run_id or new_run_id(chash)
    ev_hash = verify_evaluator()     # refuses to run if the evaluator was modified
    ledger = ledger or Ledger()
    meta = {"source": source, "evaluator_hash": ev_hash, "session": session, "rationale": rationale,
            **_git_info()}
    ledger.start(run_id, cfg, chash, meta)
    timings: dict = {}
    try:
        ev = split.SPLITS[cfg["eval_split"]]
        start = max(split.TRAIN.start, ev.start - timedelta(days=int(365.25 * float(cfg["train"]["window_years"])) + 10))
        df, feat_cols = build_panel(cfg, start, ev.end, timings, verbose, feature_workers)
        cal = store.read_calendar(cfg["dataset"], start, ev.end)
        pred, importance = rolling_predict(cfg, df, feat_cols, cal, ev.start, ev.end, timings, verbose,
                                           train_workers)

        t0 = time.time()
        rules = load_market(cfg["market"])
        metrics = evaluate(pred, rules, cfg["target"])
        timings["eval"] = time.time() - t0
        timings["total"] = time.time() - t_all
        metrics["n_features"] = len(feat_cols)
        metrics["objective"] = objective(metrics)
        metrics["importance_top"] = dict(list(importance.items())[:15])
        metrics["timings"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()}
        ledger.finish(run_id, metrics, timings)

        if save:
            out = runs_root() / run_id
            out.mkdir(parents=True, exist_ok=True)
            pred.write_parquet(out / "predictions.parquet")
            metrics["series"]["portfolio"].write_parquet(out / "portfolio_daily.parquet")
            (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
            (out / "metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "series"},
                                                         indent=2, default=str))
        if verbose:
            print(f"[{run_id}] {headline(metrics)}")
            print(f"  timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items() if isinstance(v, float)))
        return {"run_id": run_id, "metrics": metrics, "timings": timings, "config": cfg}
    except Exception as e:
        timings["total"] = time.time() - t_all
        ledger.fail(run_id, "".join(traceback.format_exception(e)), timings)
        raise
