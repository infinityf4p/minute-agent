"""Evaluator: factor values over a period and the metrics that decide admission.

Every metric is computed on eligible stock-days (base panel mask) of the requested periods only.
  rank IC      daily Spearman correlation of factor and label
  residual IC  the factor's cross-sectional rank is regressed on the style ranks (size, today's and past
               returns, volatility, turnover) each day; the residual's correlation with the label rank
  turnover     1 - mean day-to-day rank autocorrelation of the factor per stock
  long-short   top minus bottom decile label (oriented by the residual IC sign), gross and after a rough cost
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from minute_agent.factormine import compute, data, dsl
from minute_agent.factormine.config import COST_BP_ONE_WAY, THRESHOLDS, cache_dir

KEYS = ["symbol", "date"]

REFERENCE_NODES = {
    "ref_late30_volshare": {"agg": "share", "x": "volume", "win": "last30", "cond": None},
    "ref_skew": {"agg": "skew", "x": "ret", "win": "all", "cond": None},
    "ref_rv": {"agg": "std", "x": "ret", "win": "all", "cond": None},
    "ref_pv_corr": {"agg": "corr", "x": "cumret", "y": "volume", "win": "all", "cond": None},
    "ref_vwapdev": {"agg": "last", "x": "vwapdev", "win": "all", "cond": None},
    "ref_open30_ret": {"agg": "last", "x": "cumret", "win": "open30", "cond": None},
    "ref_last30_ret": {"agg": "sum", "x": "ret", "win": "last30", "cond": None},
    "ref_highvol_ret": {"agg": "mean", "x": "ret", "win": "all", "cond": "highvol"},
    "ref_auction_share": {"agg": "share", "x": "volume", "win": "auction", "cond": None},
    "ref_bigmove_frac": {"agg": "count_frac", "win": "all", "cond": "bigmove"},
}


class SpecError(ValueError):
    pass


def units_of(spec: dict) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """Units to compute and a map variable name -> (unit key, column)."""
    if spec.get("code"):
        u = compute.code_unit(spec["code"])
        return [u], {c: (u["key"], c) for c in spec["_code_cols"]}
    units, var = [], {}
    for name, node in spec["intraday"].items():
        u = compute.node_unit(node)
        units.append(u)
        var[name] = (u["key"], "v")
    return units, var


def validate(spec: dict, cfg: dict) -> dict:
    """Check a proposed factor; returns a normalized spec (raises SpecError)."""
    if not isinstance(spec, dict):
        raise SpecError("each factor must be an object")
    out = {"name": str(spec.get("name") or "f")[:40], "mechanism": str(spec.get("mechanism") or ""),
           "hypothesis": str(spec.get("hypothesis") or "")[:400]}
    try:
        if spec.get("code"):
            if spec.get("intraday"):
                raise SpecError("give either code or intraday, not both")
            out["code"] = str(spec["code"])
            try:
                cols = compute.code_columns(cfg, out["code"])
            except Exception as e:
                raise SpecError(f"code failed on a sample: {type(e).__name__}: {e}"[:600]) from e
            out["_code_cols"] = cols
            names = set(cols)
        else:
            intr = spec.get("intraday")
            if not isinstance(intr, dict) or not 1 <= len(intr) <= 3:
                raise SpecError("intraday must be an object with 1..3 named nodes")
            bad = [k for k in intr if not str(k).isidentifier() or k in data.DAILY_VARS]
            if bad:
                raise SpecError(f"bad intraday output names {bad}")
            out["intraday"] = {k: dsl.check_node(v) for k, v in intr.items()}
            names = set(intr)
        comb = dsl.check_expr(spec.get("combine"), names | set(data.DAILY_VARS))
        if not dsl.vars_used(comb) & names:
            raise SpecError("combine must use at least one intraday output")
        out["combine"] = comb
    except dsl.DSLError as e:
        raise SpecError(str(e)) from e
    return out


def spec_key(spec: dict) -> str:
    core = {"code": spec["code"]} if spec.get("code") else {"intraday": spec["intraday"]}
    return dsl.h({**core, "combine": spec["combine"]})


def factor_frame(cfg: dict, spec: dict, periods: list[tuple[date, date]],
                 computed: dict | None = None) -> pl.DataFrame:
    """Eligible stock-days of the periods with columns f, label, label_on and the styles."""
    start, end = min(a for a, _ in periods), max(b for _, b in periods)
    warm = data.shift_months(start, cfg["warmup_months"])
    units, var = units_of(spec)
    vals = computed if computed is not None else compute.compute(cfg, units, warm, end)
    fr = data.base_panel(cfg, warm, end)
    for u in units:
        v = vals[u["key"]]
        if isinstance(v, Exception):
            raise v
    for name, (key, col) in var.items():
        fr = fr.join(vals[key].select(KEYS + [pl.col(col).alias(name)]), on=KEYS, how="left")
    fr = dsl.evaluate(fr.sort("symbol", "date"), spec["combine"])
    inside = pl.lit(False)
    for a, b in periods:
        inside = inside | pl.col("date").is_between(a, b)
    return fr.filter(inside & pl.col("mask")).select(KEYS + ["f", "label", "label_on"] + data.STYLES)


def _ranked(fr: pl.DataFrame) -> pl.DataFrame:
    """Centered cross-sectional ranks in [-0.5, 0.5] (nulls kept as null)."""
    def rk(c):
        return (pl.col(c).rank() / pl.col(c).count() - 0.5).over("date")
    return fr.with_columns([rk(c).alias(f"r_{c}") for c in ["f", "label", "label_on"] + data.STYLES])


def _residual_ic(df: pl.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Per date: corr(residual of r_f on style ranks, label rank). Returns (dates, ic)."""
    df = df.sort("date")
    dates = df["date"].to_numpy()
    f = df["r_f"].to_numpy()
    y = df[f"r_{label}"].to_numpy()
    X = np.column_stack([df[f"r_{s}"].fill_null(0.0).to_numpy() for s in data.STYLES])
    cuts = np.flatnonzero(np.diff(dates.astype("datetime64[D]").astype(np.int64))) + 1
    out_d, out_ic = [], []
    for idx in np.split(np.arange(len(dates)), cuts):
        if len(idx) < THRESHOLDS["min_stocks_per_date"]:
            continue
        A = np.column_stack([np.ones(len(idx)), X[idx]])
        beta, *_ = np.linalg.lstsq(A, f[idx], rcond=None)
        e = f[idx] - A @ beta
        yy = y[idx]
        if e.std() < 1e-12 or yy.std() < 1e-12:
            continue
        out_d.append(dates[idx[0]])
        out_ic.append(float(np.corrcoef(e, yy)[0, 1]))
    return np.array(out_d), np.array(out_ic)


def _stats(ic: np.ndarray) -> dict:
    n = len(ic)
    if n < 20:
        return {"mean": float("nan"), "ir": float("nan"), "t": float("nan"), "n": n}
    m, s = float(ic.mean()), float(ic.std(ddof=1))
    return {"mean": m, "ir": m / s if s > 0 else float("nan"), "t": m / s * np.sqrt(n) if s > 0 else float("nan"), "n": n}


def metrics(fr: pl.DataFrame, periods: list[tuple[date, date]], sign: float | None = None) -> tuple[dict, pl.DataFrame]:
    """Metrics of a factor frame; also returns the oriented centered rank (symbol, date, o) for redundancy.
    sign: orientation fixed elsewhere (validation uses the mining sign); None = from this frame's residual IC."""
    total = fr.height
    ok = fr.filter(pl.col("f").is_not_null())
    cover = ok.height / total if total else 0.0
    m: dict = {"coverage": cover}
    if ok.height == 0 or ok["f"].n_unique() < 3:
        m["error"] = "factor is constant or empty"
        return m, pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date, "o": pl.Float64})
    r = _ranked(ok)
    ic_d = (r.group_by("date").agg(pl.corr("r_f", "r_label").alias("ic"), pl.len().alias("n"))
            .filter(pl.col("n") >= THRESHOLDS["min_stocks_per_date"]).sort("date"))
    raw = _stats(ic_d["ic"].drop_nans().drop_nulls().to_numpy())
    dts, res = _residual_ic(r, "label")
    _, res_on = _residual_ic(r, "label_on")
    rs = _stats(res)
    s = sign if sign is not None else (float(np.sign(rs["mean"])) if np.isfinite(rs["mean"]) and rs["mean"] != 0 else 1.0)
    folds = data.fold_of(dts.astype("datetime64[D]"), periods)
    fold_means = [float(res[folds == i].mean()) if (folds == i).sum() >= 20 else float("nan") for i in range(len(periods))]
    m.update({"ic": raw["mean"], "icir": raw["ir"], "res_ic": rs["mean"], "res_ir": rs["ir"], "res_t": rs["t"],
              "n_days": rs["n"], "res_ic_folds": fold_means, "res_ic_overnight": _stats(res_on)["mean"], "sign": s})
    # turnover: day-to-day rank autocorrelation per stock
    r = r.sort("symbol", "date").with_columns(pl.col("r_f").shift(1).over("symbol").alias("r_prev"))
    ac = r.group_by("date").agg(pl.corr("r_f", "r_prev").alias("ac")).drop_nulls()["ac"].drop_nans()
    m["rank_autocorr"] = float(ac.mean()) if len(ac) else float("nan")
    # long-short deciles, oriented
    r = r.with_columns((s * pl.col("r_f")).alias("o"))
    q = r.with_columns(pl.when(pl.col("o") >= 0.4).then(1).when(pl.col("o") < -0.4).then(-1).otherwise(0).alias("q"))
    ls = (q.filter(pl.col("q") != 0).group_by("date", "q").agg(pl.col("label").mean().alias("y"))
          .pivot(on="q", index="date", values="y").drop_nulls())
    if ls.height:
        gross = (ls["1"] - ls["-1"]).to_numpy() * 1e4
        q = q.sort("symbol", "date").with_columns(pl.col("q").shift(1).over("symbol").alias("q_prev"))
        chg = (q.filter(pl.col("q") != 0).group_by("date", "q")
               .agg((pl.col("q_prev") != pl.col("q")).fill_null(True).mean().alias("c"))
               .group_by("date").agg(pl.col("c").sum()))
        cost = 2 * float(chg["c"].mean()) * COST_BP_ONE_WAY
        m.update({"ls_gross_bp": float(gross.mean()), "ls_net_bp": float(gross.mean() - cost),
                  "ls_cost_bp": cost})
    return m, r.select(KEYS + ["o"])


def corr_with(o: pl.DataFrame, others: pl.DataFrame) -> dict[str, float]:
    """Time-averaged cross-sectional correlation of oriented ranks with each column of `others`."""
    cols = [c for c in others.columns if c not in KEYS]
    if not cols:
        return {}
    j = o.join(others, on=KEYS, how="inner")
    per = j.group_by("date").agg([pl.corr("o", c).alias(c) for c in cols])
    return {c: float(per[c].drop_nans().drop_nulls().mean() or 0.0) for c in cols}


def reference_frame(cfg: dict, periods: list[tuple[date, date]]) -> pl.DataFrame:
    """Centered ranks of classic intraday factors and the daily variables, for the novelty measure."""
    start, end = min(a for a, _ in periods), max(b for _, b in periods)
    p = cache_dir(cfg) / f"reference_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    out = None
    for name, node in REFERENCE_NODES.items():
        spec = {"intraday": {"a": node}, "combine": {"var": "a"}}
        fr = factor_frame(cfg, spec, periods).filter(pl.col("f").is_not_null())
        col = fr.select(KEYS + [((pl.col("f").rank() / pl.col("f").count()).over("date") - 0.5).alias(name)])
        out = col if out is None else out.join(col, on=KEYS, how="full", coalesce=True)
    base = data.base_panel(cfg, start, end).filter(pl.col("mask"))
    dv = base.select(KEYS + [((pl.col(c).rank() / pl.col(c).count()).over("date") - 0.5).alias(f"ref_{c}")
                             for c in data.DAILY_VARS])
    out = out.join(dv, on=KEYS, how="full", coalesce=True)
    out.write_parquet(p)
    return out
