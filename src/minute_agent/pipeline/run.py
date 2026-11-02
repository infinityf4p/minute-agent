"""Run one v2 workflow config: features -> target -> rolling model -> template backtest -> metrics.

Phases (which data a run may see):
    screen  one fold, train ... 2020, evaluate 2021          (cheap first pass, agent-visible)
    folds   evaluate 2020 and 2021, train only on earlier data (agent-visible feedback)
    valid   evaluate 2022-2023                                (evaluator only: final selection)
    test    evaluate 2024-01 .. 2026-10                       (only inside final_test_unlock)
Reads never go past the phase end, so a target that needs later data is dropped
(e.g. the last days of 2021 in the folds phase never peek into 2022).
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from minute_agent import split
from minute_agent import panel as P
from minute_agent.backtest.base_universe import base_positions
from minute_agent.backtest.book import densify, simulate_book
from minute_agent.backtest.costs import CostModel
from minute_agent.backtest.exec_table import auction_table, exec_table
from minute_agent.backtest.t0 import simulate_t0
from minute_agent.data import store
from minute_agent.data.schema import BAR_TIMES, bar_index_at
from minute_agent.evaluator.integrity import verify as verify_kernel
from minute_agent.experiment.ledger import Ledger
from minute_agent.experiment.runner import BASE_COLS, DEFAULT_FEATURE_WORKERS, _git_info, retrain_dates
from minute_agent.features.cache import FeatureCache
from minute_agent.features.extra_ops import seq_columns
from minute_agent.market import load_market
from minute_agent.models import SEQ_MODEL_TYPES, cpu_count, make_model
from minute_agent.pipeline import spec
from minute_agent.paths import runs_root

PHASES = {
    "screen": [(date(2021, 1, 1), date(2021, 12, 31))],
    "folds": [(date(2020, 1, 1), date(2020, 12, 31)), (date(2021, 1, 1), date(2021, 12, 31))],
    "valid": [(split.VALID.start, split.VALID.end)],
    "test": [(split.LOCKED_TEST.start, split.LOCKED_TEST.end)],
}
DATA_START = date(2016, 1, 1)
FLATTEN_BAR = bar_index_at("cn", spec.FLATTEN_AT)
FLATTEN_RETRY = [FLATTEN_BAR, bar_index_at("cn", "14:50"), bar_index_at("cn", "14:55")]
POOL_RANGE = {"dev": (date(2016, 1, 1), date(2023, 12, 31)), "test": (date(2024, 1, 1), date(2026, 10, 31))}


def hhmm(bidx: int) -> str:
    return BAR_TIMES["cn"][bidx].strftime("%H:%M")


def config_hash(cfg: dict, settings: spec.Settings) -> str:
    blob = json.dumps({"cfg": {k: v for k, v in cfg.items() if k != "name"}, "settings": settings.as_dict()},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# data pieces


def base_pool(settings: spec.Settings, which: str) -> tuple[list[str], str]:
    lo, hi = POOL_RANGE[which]
    b = base_positions(settings.dataset, lo, hi, n=settings.base_n, capital=settings.capital)
    syms = sorted(b["symbol"].unique().to_list())
    tag = f"pool{settings.base_n}-{which}-" + hashlib.sha1(",".join(syms).encode()).hexdigest()[:8]
    return syms, tag


def load_features(settings, specs, bars, start, end, symbols, tag, timings, workers) -> pl.DataFrame:
    frames = []
    t0 = time.time()
    for d in bars:
        fc = FeatureCache(settings.dataset, "cn", hhmm(d), symbols=symbols, universe_tag=tag)
        fc.build(specs, start, end, workers=workers)
        x = fc.load(specs, start, end).with_columns(pl.lit(d, dtype=pl.Int16).alias("d"))
        if symbols is not None:
            x = x.filter(pl.col("symbol").is_in(symbols))
        frames.append(x)
    timings["features"] = timings.get("features", 0.0) + time.time() - t0
    return pl.concat(frames, how="diagonal_relaxed")


def make_labels(kind: dict, ex: pl.DataFrame, auc: pl.DataFrame, daily: pl.DataFrame, cal: list[date]) -> pl.DataFrame:
    """Forward return from the executable price after the decision bar (ex.px)."""
    k = kind["kind"]
    adj = daily.select("symbol", "date", "close", "adj_factor")
    base = ex.select("symbol", "date", "d", "px").join(adj, on=["symbol", "date"], how="left")
    if k == "fwd_days":
        n = int(kind["days"])
        shift = pl.DataFrame({"date": cal[:-n], "td": cal[n:]}, schema={"date": pl.Date, "td": pl.Date})
        fut = adj.rename({"date": "td", "close": "f_close", "adj_factor": "f_adj"})
        lab = base.join(shift, on="date").join(fut, on=["symbol", "td"], how="left").with_columns(
            (pl.col("f_close") * pl.col("f_adj") / (pl.col("px") * pl.col("adj_factor")) - 1).alias("label"))
    elif k == "overnight":
        shift = pl.DataFrame({"date": cal[:-1], "td": cal[1:]}, schema={"date": pl.Date, "td": pl.Date})
        fut = auc.select("symbol", pl.col("date").alias("td"), "open_px").join(
            adj.select("symbol", pl.col("date").alias("td"), pl.col("adj_factor").alias("f_adj")), on=["symbol", "td"])
        lab = base.join(shift, on="date").join(fut, on=["symbol", "td"], how="left").with_columns(
            (pl.col("open_px") * pl.col("f_adj") / (pl.col("px") * pl.col("adj_factor")) - 1).alias("label"))
    elif k in ("intraday", "to_flatten"):
        pts = sorted(ex["d"].unique().to_list())
        if k == "intraday":
            h = int(kind["minutes"])
            target_of = {d: min([p for p in pts if p >= min(d + h, FLATTEN_BAR)] or [FLATTEN_BAR]) for d in pts}
        else:
            target_of = {d: FLATTEN_BAR for d in pts}
        m = pl.DataFrame({"d": list(target_of), "td": list(target_of.values())}, schema={"d": pl.Int16, "td": pl.Int16})
        fut = ex.select("symbol", "date", pl.col("d").alias("td"), pl.col("px").alias("f_px"))
        lab = base.join(m, on="d").join(fut, on=["symbol", "date", "td"], how="left").with_columns(
            pl.when(pl.col("td") > pl.col("d")).then(pl.col("f_px") / pl.col("px") - 1).alias("label"))
    else:
        raise ValueError(k)
    return lab.select("symbol", "date", "d", "label")


def norm_label(df: pl.DataFrame, how: str) -> pl.DataFrame:
    g = ["date", "d"]
    if how == "rank":
        y = (pl.col("label").rank().over(g) - 1) / (pl.col("label").count().over(g) - 1).clip(lower_bound=1) - 0.5
    elif how == "raw":
        y = pl.col("label")
    elif how == "vol":
        y = (pl.col("label") / pl.col("sigma").clip(lower_bound=1e-4)).clip(-50, 50)
    else:
        raise ValueError(how)
    return df.with_columns(y.alias("y"))


def build_panel(cfg, settings, role, bars, start, end, timings, workers, symbols=None, tag=None):
    """Rows (symbol, date, d) with normalized features, label, y, tradability; plus exec/auction/daily."""
    target = cfg["target"] if role == "book" or cfg["template"] == "T2" else cfg["t0_target"]
    specs = list(cfg["features"])
    if cfg["model"]["type"] in SEQ_MODEL_TYPES and not any(f["op"] == "seq_bars" for f in specs):
        specs.append({"op": "seq_bars", "bucket_min": 10})
    X = load_features(settings, specs, bars, start, end, symbols, tag, timings, workers)
    t0 = time.time()
    style = cfg["execution"]["style"]
    # one execution table per style covers every allowed decision bar, so it is built once
    if role == "book":
        ex_bars = [bar_index_at("cn", h) for h in spec.DECISION_TIMES]
    else:
        ex_bars = sorted({bar_index_at("cn", h) for g in spec.T0_GRIDS.values() for h in g}
                         | set(FLATTEN_RETRY) | {bar_index_at("cn", spec.T0_OVERNIGHT_DECISION)})
    ex = exec_table(settings.dataset, start, end, ex_bars, style, symbols=symbols, universe_tag=tag or "all")
    auc = auction_table(settings.dataset, start, end, symbols=symbols, universe_tag=tag or "all")
    cal = store.read_calendar(settings.dataset, start - timedelta(days=400), end)
    daily = store.read_daily(settings.dataset, start - timedelta(days=60), end, symbols=symbols)
    sec = store.read_security(settings.dataset)
    ex = ex.filter(pl.col("d").is_in(sorted(set(bars) | (set(FLATTEN_RETRY) if role == "t0" else set()))))
    lab = make_labels(target, ex.filter(pl.col("d").is_in(bars)) if target["kind"] not in ("intraday", "to_flatten")
                      else ex, auc, daily, cal)
    lab = lab.filter(pl.col("d").is_in(bars))
    feat_cols = [c for c in X.columns if c not in BASE_COLS and c not in ("symbol", "date", "d")]
    rules = load_market("cn")
    unis = []
    for d in bars:
        xb = X.filter(pl.col("d") == d)
        u = P.universe(daily, sec, cal, xb, rules.universe, cfg.get("universe_filter"))
        unis.append(u.with_columns(pl.lit(d, dtype=pl.Int16).alias("d")))
    uni = pl.concat(unis).rename({"buyable": "buyable_now"})
    df = (X.select(["symbol", "date", "d", *feat_cols])
          .join(uni, on=["symbol", "date", "d"], how="inner")
          .join(lab, on=["symbol", "date", "d"], how="left")
          .join(ex.select("symbol", "date", "d", "sigma"), on=["symbol", "date", "d"], how="left"))
    if role == "book":
        df = df.filter(pl.col("in_universe"))
    tab = [c for c in feat_cols if c not in set(seq_columns(feat_cols))]
    if len(bars) > 1:
        df = df.with_columns((pl.col("d").cast(pl.Float64) / 240).alias("tod"))
        tab, feat_cols = tab + ["tod"], feat_cols + ["tod"]
    df = P.cross_section_norm(df.rename({"date": "_date"}).with_columns(
        (pl.col("_date").cast(pl.Int64) * 1000 + pl.col("d").cast(pl.Int64)).alias("date")), tab,
        cfg["cross_section_norm"]).drop("date").rename({"_date": "date"})
    df = norm_label(df, cfg["label_norm"]).sort("date", "d", "symbol")
    timings["panel"] = timings.get("panel", 0.0) + time.time() - t0
    return df, feat_cols, ex, auc, daily, cal, target


def rolling_predict(cfg, df, feat_cols, cal, ev_start, ev_end, horizon_days, timings):
    t0 = time.time()
    pts = retrain_dates(ev_start, ev_end, cfg["train"]["retrain_every"])
    bounds = pts[1:] + [ev_end + timedelta(days=1)]
    window = timedelta(days=int(round(365.25 * float(cfg["train"]["window_years"]))))
    cal_arr = np.array(cal, dtype="datetime64[D]")
    emb = max(1, int(horizon_days)) + 1
    model0 = make_model(cfg["model"])
    threads = cpu_count() if model0.device == "cuda" else max(1, cpu_count())
    preds = []
    for p, nxt in zip(pts, bounds):
        k = int(np.searchsorted(cal_arr, np.datetime64(p)))
        tr_end = cal[max(0, k - emb)]
        tr = df.filter(pl.col("date").is_between(max(p - window, DATA_START), tr_end, closed="left") & pl.col("y").is_not_null()
                       & pl.col("y").is_finite())
        te = df.filter(pl.col("date").is_between(p, nxt, closed="left"))
        if tr.height < 1000 or te.height == 0:
            continue
        m = make_model(cfg["model"], seed=int(cfg.get("seed", 0)))
        m.set_threads(threads)
        m.fit(tr.select(feat_cols).to_numpy(), tr["y"].to_numpy(), feat_cols)
        preds.append(te.select("symbol", "date", "d", "label", "buyable_now", "in_universe")
                     .with_columns(pl.Series("score", m.predict(te.select(feat_cols).to_numpy()))))
    timings["train"] = timings.get("train", 0.0) + time.time() - t0
    return pl.concat(preds) if preds else pl.DataFrame()


def horizon_days(target: dict) -> int:
    return {"fwd_days": int(target.get("days", 1)), "overnight": 1}.get(target["kind"], 0)


# ---------------------------------------------------------------------------
# backtests


def _book_dense(pred, ex, auc, daily, bar, dates):
    syms = sorted(pred["symbol"].unique().to_list())
    e = ex.filter(pl.col("d") == bar)
    d = daily.filter(pl.col("symbol").is_in(syms)).sort("symbol", "date").with_columns(
        (pl.col("close") * pl.col("adj_factor")).alias("_ac"))
    d = d.with_columns((pl.col("_ac") / pl.col("_ac").shift(20) - 1).over("symbol").alias("r20"))
    d = d.with_columns(pl.col("r20").shift(1).over("symbol"))
    u20 = d.group_by("date").agg(pl.col("r20").mean().alias("univ_ret20"))
    d = d.join(u20, on="date", how="left")
    frames = {
        "pred": (pred.with_columns(pl.col("in_universe").cast(pl.Float64).alias("in_univ"),
                                   pl.col("buyable_now").cast(pl.Float64)), ["score", "in_univ", "buyable_now"]),
        "daily": (d.select("symbol", "date", "close", pl.col("adj_factor").alias("adj"), "univ_ret20"),
                  ["close", "adj", "univ_ret20"]),
        "ex": (e.with_columns(pl.col("can_buy").cast(pl.Float64), pl.col("can_sell").cast(pl.Float64)),
               ["px", "avail", "can_buy", "can_sell", "sigma"]),
        "auc": (auc.select("symbol", "date", "open_px", pl.col("open_can_sell").cast(pl.Float64),
                           pl.col("open_vol").alias("avail_open")), ["open_px", "open_can_sell", "avail_open"]),
    }
    dense = densify(frames, dates, syms)
    A = dense.arrays
    # forward-fill closes/adj over suspension gaps so positions keep a price
    for k in ("close", "adj"):
        a = A[k]
        for t in range(1, a.shape[0]):
            m = ~np.isfinite(a[t])
            a[t, m] = a[t - 1, m]
    A["in_univ"] = np.nan_to_num(A["in_univ"])
    A["buyable_now"] = np.nan_to_num(A["buyable_now"])
    for k in ("can_buy", "can_sell", "open_can_sell"):
        A[k] = np.nan_to_num(A[k])
    return dense


def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def _maxdd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + np.nan_to_num(r))
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else float("nan")


def backtest(cfg, settings, costs, pieces) -> tuple[pl.DataFrame, dict]:
    """Returns the daily series (one row per day) and template-specific stats."""
    tpl = cfg["template"]
    port = dict(cfg.get("portfolio") or {})
    out_stats = {}
    daily_series = None
    if tpl in ("T1", "T3", "T4"):
        pred, ex, auc, daily, bar = pieces["book"]
        dates = sorted(pred["date"].unique().to_list())
        dense = _book_dense(pred, ex, auc, daily, bar, dates)
        port["mode"] = "next_open" if tpl == "T3" else "hold"
        bk, pos = simulate_book(dense, port, costs, capital=settings.capital, position_sizing=settings.position_sizing)
        daily_series = bk.select("date", pl.col("excess_net").alias("ret"), "excess_gross", "cost", "turnover",
                                 "exposure", "n_pos", "ret_net", "equity")
        out_stats.update({"book_turnover": float(bk["turnover"].mean()), "avg_positions": float(bk["n_pos"].mean()),
                          "avg_exposure": float(bk["exposure"].mean())})
    if tpl in ("T2", "T4"):
        pred, ex, auc, daily, bar = pieces["t0"]
        sig = pred.select("symbol", "date", "d", "score")
        if tpl == "T2":
            dates = sorted(sig["date"].unique().to_list())
            base = base_positions(settings.dataset, dates[0], dates[-1], n=settings.base_n, capital=settings.capital)
        else:
            rows = []
            for d, V, age in pos:
                held = np.where((V > 0) & (age >= 1))[0]
                rows += [(dense.symbols[j], d.astype(object), float(V[j])) for j in held]
            base = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "date": pl.Date, "base_value": pl.Float64}, orient="row")
            base = base.filter(pl.col("symbol").is_in(sig["symbol"].unique()))
        if cfg["t0"]["mode"] == "intraday":
            fl = ex.filter(pl.col("d").is_in(FLATTEN_RETRY)).select("symbol", "date", "d").with_columns(pl.lit(None, pl.Float64).alias("score"))
            sig = pl.concat([sig.filter(pl.col("d") < FLATTEN_BAR), fl], how="diagonal_relaxed")
        params = {**cfg["t0"], "flatten_at": FLATTEN_BAR}
        ov, st = simulate_t0(base, sig, ex, auc, params, costs, position_sizing=settings.position_sizing)
        out_stats.update({f"t0_{k}": v for k, v in st.items()})
        if tpl == "T2":
            daily_series = ov.select("date", pl.col("overlay_net").alias("ret"), pl.col("overlay_gross").alias("excess_gross"),
                                     pl.col("overlay_cost").alias("cost"))
        else:
            add = ov.select("date", "net")
            daily_series = daily_series.join(add, on="date", how="left").with_columns(
                (pl.col("ret") + pl.col("net").fill_null(0.0) / pl.col("equity").shift(1).fill_null(pl.col("equity"))).alias("ret")
            ).drop("net")
            out_stats["t0_overlay_bp_per_day_on_equity"] = float(
                (ov["net"].sum() / daily_series["equity"].mean()) / max(daily_series.height, 1) * 1e4)
    return daily_series, out_stats


def summarize(tpl: str, series: pl.DataFrame) -> dict:
    r = series["ret"].to_numpy()
    m = {"days": len(r), "sharpe": _sharpe(r), "ann_ret": float(np.nanmean(r) * 252), "maxdd": _maxdd(r),
         "mean_bp_per_day": float(np.nanmean(r) * 1e4), "hit": float(np.nanmean(r > 0)),
         "cost_bp_per_day": float(np.nanmean(series["cost"].to_numpy()) * 1e4)}
    if "turnover" in series.columns:
        m["turnover"] = float(series["turnover"].mean())
    m["objective"] = m["mean_bp_per_day"] if tpl == "T2" else m["sharpe"]
    return m


def ic_stats(pred: pl.DataFrame) -> dict:
    p = pred.filter(pl.col("label").is_not_null() & pl.col("label").is_finite())
    g = p.with_columns(pl.col("score").rank().over("date", "d").alias("rs"), pl.col("label").rank().over("date", "d").alias("rl")) \
        .group_by("date", "d").agg(pl.corr("rs", "rl").alias("ric"), pl.len().alias("n")).filter(pl.col("n") >= 20)
    v = g["ric"].to_numpy()
    return {"rank_ic": float(np.nanmean(v)), "rank_icir": float(np.nanmean(v) / np.nanstd(v)) if len(v) > 1 else float("nan")}


# ---------------------------------------------------------------------------


def run_pipeline(cfg_in: dict, settings: spec.Settings, phase: str = "folds", source: str = "human",
                 session: str | None = None, rationale: str | None = None, ledger: Ledger | None = None,
                 run_id: str | None = None, verbose: bool = False, save: bool = True,
                 feature_workers: int = DEFAULT_FEATURE_WORKERS) -> dict:
    t_all = time.time()
    cfg = spec.check(cfg_in, settings)
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {list(PHASES)}")
    if phase == "test" and not split._FINAL_EVAL:
        raise split.LockedDataError("the test phase only runs inside split.final_test_unlock()")
    chash = config_hash(cfg, settings)
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S-%f}-{chash}"
    kernel = verify_kernel()
    costs = CostModel.load(settings.costs)
    ledger = ledger or Ledger()
    record = {"cfg": cfg, "dataset": settings.dataset, "market": "cn", "eval_split": phase,
              "target": json.dumps(cfg.get("target") or cfg.get("t0_target")), "template": cfg["template"],
              "settings": settings.as_dict(), "costs": {"name": costs.name, "hash": costs.hash(), **costs.raw}}
    ledger.start(run_id, record, chash, {"source": source, "evaluator_hash": kernel, "session": session,
                                         "rationale": rationale, **_git_info()})
    timings: dict = {}
    try:
        periods = PHASES[phase]
        ev_lo, ev_hi = periods[0][0], periods[-1][1]
        data_start = max(DATA_START, ev_lo - timedelta(days=int(365.25 * cfg["train"]["window_years"]) + 30))
        bars = spec.decision_bars(cfg)
        pieces, ic = {}, {}
        for role, b in bars.items():
            symbols = tag = None
            if role == "t0":
                symbols, tag = base_pool(settings, "test" if phase == "test" else "dev")
            df, fcols, ex, auc, daily, cal, target = build_panel(cfg, settings, role, b, data_start, ev_hi, timings,
                                                                 feature_workers, symbols, tag)
            preds = []
            for lo, hi in periods:
                preds.append(rolling_predict(cfg, df, fcols, cal, lo, hi, horizon_days(target), timings))
            pred = pl.concat([p for p in preds if p.height])
            ic[role] = ic_stats(pred)
            pieces[role] = (pred, ex, auc, daily, b[0])
        t0 = time.time()
        results = {}
        for level, cm in (("base", costs), ("x2", CostModel({**costs.raw, "cost_multiplier": 2 * costs.raw.get("cost_multiplier", 1.0)})),
                          ("gross", CostModel({**costs.raw, **{k: 0.0 for k in ("commission_rate", "min_commission",
                                                                                "transfer_fee_rate", "slippage_bps", "impact_coef")},
                                               "stamp_duty_sell": 0.0}))):
            series, st = backtest(cfg, settings, cm, pieces)
            results[level] = (series, st)
        timings["backtest"] = time.time() - t0
        series, st = results["base"]
        metrics = summarize(cfg["template"], series)
        metrics.update({f"x2_{k}": v for k, v in summarize(cfg["template"], results["x2"][0]).items() if k in ("sharpe", "mean_bp_per_day", "objective")})
        metrics.update({f"gross_{k}": v for k, v in summarize(cfg["template"], results["gross"][0]).items() if k in ("sharpe", "mean_bp_per_day", "objective")})
        metrics.update(st)
        for role, v in ic.items():
            metrics.update({f"{role}_{k}": x for k, x in v.items()})
        if len(periods) > 1:
            for i, (lo, hi) in enumerate(periods):
                sub = series.filter(pl.col("date").is_between(lo, hi))
                metrics[f"fold{i + 1}_objective"] = summarize(cfg["template"], sub)["objective"]
        metrics["rank_ic"] = metrics.get("book_rank_ic", metrics.get("t0_rank_ic"))
        metrics["ic"] = metrics["rank_ic"]
        metrics["excess_net_bp"] = metrics["mean_bp_per_day"]
        metrics["excess_net_ann"] = metrics["ann_ret"]
        metrics["excess_net_sharpe"] = metrics["sharpe"]
        metrics["top_turnover"] = metrics.get("turnover")
        timings["total"] = time.time() - t_all
        metrics["timings"] = {k: round(v, 1) for k, v in timings.items()}
        ledger.finish(run_id, {**metrics, "objective": metrics["objective"]}, timings)
        if save:
            out = runs_root() / run_id
            out.mkdir(parents=True, exist_ok=True)
            series.write_parquet(out / "daily.parquet")
            (out / "config.json").write_text(json.dumps(record, indent=1, default=str))
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
        if verbose:
            print(f"[{run_id}] {cfg['template']} obj={metrics['objective']:.3f} sharpe={metrics['sharpe']:.2f} "
                  f"{metrics['mean_bp_per_day']:.2f}bp/d rankIC={metrics['rank_ic']:.4f} {timings['total']:.0f}s", flush=True)
        return {"run_id": run_id, "metrics": metrics, "timings": timings, "config": cfg}
    except Exception as e:
        timings["total"] = time.time() - t_all
        ledger.fail(run_id, "".join(traceback.format_exception(e)), timings)
        raise
