"""Minute -> daily feature operators (PLAN.md section 4, v1 action space).

Every operator has the signature

    fn(m: pl.DataFrame, daily: pl.DataFrame, d: int, **params) -> pl.DataFrame

* `m` holds minute bars of many symbols and days, ALREADY cut at the decision
  bar (bidx <= d), sorted by symbol, date, bidx, with an extra column `lr`
  (1-minute log return within the day; null on the first bar).
* `daily` holds daily bars (standard schema). It contains rows for dates >= the
  feature date, so operators must lag daily fields themselves; the leakage
  checker (features.leakage) verifies that they do.
* The output has one row per (symbol, date) present in `m`.

Windows are in bars (trading minutes), counted back from the decision bar and
clipped at the first bar of the day. Lunch break is skipped automatically.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

KEYS = ["symbol", "date"]

OpFn = Callable[..., pl.DataFrame]
REGISTRY: dict[str, tuple[OpFn, str]] = {}   # name -> (fn, version)


def register(name: str, version: str = "1"):
    def deco(fn: OpFn) -> OpFn:
        REGISTRY[name] = (fn, version)
        return fn
    return deco


def prepare_minute(m: pl.DataFrame) -> pl.DataFrame:
    """Sort and add the 1-minute log return used by most operators."""
    m = m.sort(["symbol", "date", "bidx"])
    return m.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().over(["symbol", "date"]).alias("lr")
    )


def _close_at_or_before(b: int) -> pl.Expr:
    if b < 0:
        return pl.col("close").first()
    return pl.col("close").filter(pl.col("bidx") <= b).last()


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b != 0).then(a / b).otherwise(None)


@register("base")
def op_base(m: pl.DataFrame, daily: pl.DataFrame, d: int) -> pl.DataFrame:
    """Bookkeeping columns used by the universe filter, not model features by default."""
    return m.group_by(KEYS).agg(
        pl.col("close").last().alias("price_dec"),
        pl.col("amount").sum().alias("amt_dec"),
        pl.col("bidx").last().alias("last_bidx"),
    )


@register("ret")
def op_ret(m, daily, d, window_min=(5, 30, 60, 240)):
    return m.group_by(KEYS).agg([
        (pl.col("close").last() / _close_at_or_before(d - w)).log().alias(f"ret_{w}")
        for w in window_min
    ])


@register("realized_vol")
def op_realized_vol(m, daily, d, window_min=(30, 240)):
    return m.group_by(KEYS).agg([
        pl.col("lr").filter(pl.col("bidx") > d - w).pow(2).sum().sqrt().alias(f"rv_{w}")
        for w in window_min
    ])


@register("vwap_dev")
def op_vwap_dev(m, daily, d, at="decision_time"):
    if at != "decision_time":
        raise ValueError("vwap_dev only supports at=decision_time")
    vwap = _safe_div(pl.col("amount").sum(), pl.col("volume").sum())
    return m.group_by(KEYS).agg((pl.col("close").last() / vwap - 1).alias("vwap_dev"))


def _bucket_mask(name: str, d: int) -> pl.Expr:
    """open30 = first 30 minutes incl. the opening auction; last30 = 30 bars up to the
    decision bar (NOT the last 30 minutes of the session); mid = the rest."""
    b = pl.col("bidx")
    if name.startswith("open"):
        return b <= int(name[4:])
    if name.startswith("last"):
        return b > d - int(name[4:])
    if name == "mid":
        return (b > 30) & (b <= d - 30)
    raise ValueError(f"unknown volume bucket {name!r}")


@register("volume_profile")
def op_volume_profile(m, daily, d, buckets=("open30", "mid", "last30")):
    tot = pl.col("volume").sum()
    return m.group_by(KEYS).agg([
        _safe_div(pl.col("volume").filter(_bucket_mask(b, d)).sum(), tot).alias(f"vp_{b}")
        for b in buckets
    ])


@register("last_n_min_stats")
def op_last_n_min_stats(m, daily, d, n=30):
    win = pl.col("bidx") > d - n
    p = f"l{n}"
    return m.group_by(KEYS).agg(
        (pl.col("close").last() / _close_at_or_before(d - n)).log().alias(f"{p}_ret"),
        pl.col("lr").filter(win).std().alias(f"{p}_vol"),
        _safe_div(pl.col("volume").filter(win).sum(), pl.col("volume").sum()).alias(f"{p}_vshare"),
        ((pl.col("high").filter(win).max() - pl.col("low").filter(win).min()) / pl.col("close").last()).alias(f"{p}_range"),
        (pl.col("lr").filter(win) > 0).mean().alias(f"{p}_upfrac"),
        _safe_div(pl.col("lr").filter(win).abs().sum(), pl.col("amount").filter(win).sum() / 1e8).alias(f"{p}_illiq"),
    )


@register("jump_count")
def op_jump_count(m, daily, d, threshold_sigma=4):
    k = float(threshold_sigma)
    return m.group_by(KEYS).agg(
        (pl.col("lr").abs() > k * pl.col("lr").std()).sum().cast(pl.Float64).alias(f"jump_{threshold_sigma}s")
    )


@register("daily_lag")
def op_daily_lag(m, daily, d, fields=("ret", "turnover"), lags=(1, 5, 20)):
    """Daily history ending the day BEFORE the feature date (shift(1) per symbol)."""
    if any(int(l) < 1 for l in lags):
        raise ValueError("daily_lag needs lags >= 1")
    dl = daily.sort(["symbol", "date"]).with_columns(
        (pl.col("close") * pl.col("adj_factor")).log().diff().over("symbol").alias("_lret")
    )
    exprs = []
    for f in fields:
        for L in lags:
            if f == "ret":
                e = pl.col("_lret").rolling_sum(int(L), min_samples=1)
            elif f == "turnover":
                e = pl.col("turnover").rolling_mean(int(L), min_samples=1)
            else:
                raise ValueError(f"daily_lag: unsupported field {f!r}")
            exprs.append(e.shift(1).over("symbol").alias(f"dl_{f}_{L}"))
    out = dl.with_columns(exprs).select(KEYS + [x.meta.output_name() for x in exprs])
    keys = m.select(KEYS).unique()
    return keys.join(out, on=KEYS, how="left")


@register("return_moments")
def op_return_moments(m, daily, d, window_min=(240,)):
    """Skewness and kurtosis of 1-minute returns over the last w bars."""
    out = []
    for w in window_min:
        win = pl.col("bidx") > d - w
        out += [pl.col("lr").filter(win).skew().alias(f"skew_{w}"),
                pl.col("lr").filter(win).kurtosis().alias(f"kurt_{w}")]
    return m.group_by(KEYS).agg(out)


@register("pv_corr")
def op_pv_corr(m, daily, d, window_min=(240,)):
    """Correlation of |1-min return| and volume, and of return and volume, over the last w bars."""
    out = []
    for w in window_min:
        win = pl.col("bidx") > d - w
        out += [pl.corr(pl.col("lr").filter(win).abs(), pl.col("volume").filter(win)).alias(f"pvabs_{w}"),
                pl.corr(pl.col("lr").filter(win), pl.col("volume").filter(win)).alias(f"pvsgn_{w}")]
    return m.group_by(KEYS).agg(out)


@register("price_position")
def op_price_position(m, daily, d):
    """Where the decision price sits in today's range so far, and today's range itself."""
    hi, lo, c = pl.col("high").max(), pl.col("low").min(), pl.col("close").last()
    return m.group_by(KEYS).agg(
        _safe_div(c - lo, hi - lo).alias("pos_in_range"),
        ((hi - lo) / c).alias("day_range"),
        (c / pl.col("open").first() - 1).alias("ret_from_open"),
    )


@register("signed_volume")
def op_signed_volume(m, daily, d, window_min=(30, 240)):
    """Volume-weighted direction: sum(volume * sign(r)) / sum(volume) over the last w bars."""
    out = []
    for w in window_min:
        win = pl.col("bidx") > d - w
        out.append(_safe_div((pl.col("volume") * pl.col("lr").sign()).filter(win).sum(),
                             pl.col("volume").filter(win).sum()).alias(f"sv_{w}"))
    return m.group_by(KEYS).agg(out)


@register("daily_stats")
def op_daily_stats(m, daily, d, windows=(5, 20, 60)):
    """Daily-bar history up to the day before: volatility, illiquidity, distance to range."""
    dl = daily.sort(["symbol", "date"]).with_columns(
        (pl.col("close") * pl.col("adj_factor")).alias("_ac"),
    ).with_columns(
        pl.col("_ac").log().diff().over("symbol").alias("_lr"),
    )
    exprs = []
    for w in windows:
        w = int(w)
        exprs += [
            pl.col("_lr").rolling_std(w, min_samples=max(2, w // 2)).shift(1).over("symbol").alias(f"ds_vol_{w}"),
            (pl.col("_lr").abs() / (pl.col("amount") / 1e8)).replace([float("inf")], None)
            .rolling_mean(w, min_samples=max(1, w // 2)).shift(1).over("symbol").alias(f"ds_illiq_{w}"),
            (pl.col("_ac") / pl.col("_ac").rolling_max(w, min_samples=1) - 1).shift(1).over("symbol").alias(f"ds_offhigh_{w}"),
            (pl.col("_ac") / pl.col("_ac").rolling_min(w, min_samples=1) - 1).shift(1).over("symbol").alias(f"ds_offlow_{w}"),
        ]
    out = dl.with_columns(exprs).select(KEYS + [x.meta.output_name() for x in exprs])
    return m.select(KEYS).unique().join(out, on=KEYS, how="left")


def compute(name: str, m_prepared: pl.DataFrame, daily: pl.DataFrame, d: int, params: dict) -> pl.DataFrame:
    fn, _ = REGISTRY[name]
    out = fn(m_prepared, daily, d, **params)
    return out.sort(KEYS)


def cutoff(m: pl.DataFrame, d: int) -> pl.DataFrame:
    """Drop every bar after the decision bar. Production code always goes through this."""
    return m.filter(pl.col("bidx") <= d)


# M3 operators (sequence buckets, custom features) register themselves on import
from minute_agent.features import extra_ops  # noqa: E402,F401
