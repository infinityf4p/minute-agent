"""Fixed evaluator. The agent may not modify anything in this package.

Input: a panel with one row per (date, symbol) in the universe:
    date, symbol, score   model output, higher = more bullish
    label                 realized forward return for the target (null if unknown,
                          e.g. suspended on the exit day)
    buyable               can be bought at the decision/close (not suspended,
                          not at the up limit at the decision time)

Signal metrics (IC, RankIC, ICIR, group returns) use rows with a label.
Portfolio metrics use every universe row; a missing label counts as 0 return
(the position could not be exited and is marked flat), which is reported as
`top_label_missing_frac`.

All returns are simple returns per holding period; "bp" = 1e-4.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from minute_agent import split

TRADING_DAYS = 252
MIN_NAMES_PER_DAY = 20

REQUIRED = ["date", "symbol", "score", "label", "buyable"]


def _check_panel(panel: pl.DataFrame, allow_locked: bool) -> pl.DataFrame:
    missing = [c for c in REQUIRED if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns {missing}")
    if panel.select(pl.struct("date", "symbol").is_duplicated().any()).item():
        raise ValueError("panel has duplicate (date, symbol) rows")
    lo, hi = panel["date"].min(), panel["date"].max()
    split.check_range(lo, hi, allow_locked)
    return panel.select(REQUIRED).filter(pl.col("score").is_not_null() & pl.col("score").is_finite())


def daily_ic(panel: pl.DataFrame) -> pl.DataFrame:
    p = panel.filter(pl.col("label").is_not_null() & pl.col("label").is_finite())
    return (
        p.with_columns(
            pl.col("score").rank().over("date").alias("_rs"),
            pl.col("label").rank().over("date").alias("_rl"),
        )
        .group_by("date")
        .agg(
            pl.len().alias("n"),
            pl.corr("score", "label").alias("ic"),
            pl.corr("_rs", "_rl").alias("rank_ic"),
        )
        .filter(pl.col("n") >= MIN_NAMES_PER_DAY)
        .sort("date")
    )


def _assign_groups(df: pl.DataFrame, n_groups: int, col: str = "group") -> pl.DataFrame:
    """Group 0 = lowest score, n_groups-1 = highest, equal counts per date (ties by order)."""
    return df.with_columns(
        (
            (pl.col("score").rank(method="ordinal").over("date") - 1)
            * n_groups
            / pl.len().over("date")
        ).floor().cast(pl.Int32).alias(col)
    )


def group_returns(panel: pl.DataFrame, n_groups: int) -> pl.DataFrame:
    p = panel.filter(pl.col("label").is_not_null() & pl.col("label").is_finite())
    p = _assign_groups(p, n_groups)
    return p.group_by("date", "group").agg(pl.col("label").mean().alias("ret")).sort("date", "group")


def _summary(x: np.ndarray, periods: int = TRADING_DAYS) -> dict:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float("nan"), "std": float("nan"), "ir": float("nan"), "t": float("nan"), "n": len(x)}
    m, s = float(x.mean()), float(x.std(ddof=1))
    return {"mean": m, "std": s, "ir": m / s if s > 0 else float("nan"),
            "t": m / s * math.sqrt(len(x)) if s > 0 else float("nan"), "n": len(x)}


def _max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + np.nan_to_num(r))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min()) if len(eq) else float("nan")


def long_top_vs_universe(panel: pl.DataFrame, rules, target: str, n_groups: int) -> tuple[dict, pl.DataFrame]:
    """A-share M1 portfolio: equal-weight long the top group of buyable names,
    benchmark = equal-weight universe. Costs only on the long leg."""
    p = panel.with_columns(pl.col("label").fill_null(0.0).fill_nan(0.0).alias("ret"))
    univ = p.group_by("date").agg(pl.col("ret").mean().alias("univ"), pl.len().alias("n_univ"))
    b = _assign_groups(p.filter(pl.col("buyable")), n_groups, "bgroup")
    top = b.filter(pl.col("bgroup") == n_groups - 1)
    members = top.group_by("date").agg(pl.col("symbol").alias("names"))
    long = top.group_by("date").agg(
        pl.col("ret").mean().alias("long"),
        pl.len().alias("n_top"),
        pl.col("label").is_null().mean().alias("missing"),
    )
    daily = long.join(univ, on="date").join(members, on="date").sort("date")

    # one-sided turnover of the top group (share of names that are new today)
    names = daily["names"].to_list()
    turn = [1.0]
    for prev, cur in zip(names[:-1], names[1:]):
        cur_s = set(cur)
        turn.append(1.0 - len(cur_s & set(prev)) / max(len(cur_s), 1))
    dates = daily["date"].to_list()
    buy = np.array([rules.buy_cost(d) for d in dates])
    sell = np.array([rules.sell_cost(d) for d in dates])
    turn = np.array(turn)
    if target == "overnight":
        traded = np.ones_like(turn)      # buy at today's close, sell at tomorrow's open: full round trip
    elif target == "next_cc":
        traded = turn                    # keep names that stay in the top group
    else:
        raise ValueError(f"unknown target {target!r}")
    cost = traded * (buy + sell)

    daily = daily.drop("names").with_columns(
        pl.Series("top_turnover", turn),
        pl.Series("cost", cost),
    ).with_columns(
        (pl.col("long") - pl.col("univ")).alias("excess_gross"),
        (pl.col("long") - pl.col("univ") - pl.col("cost")).alias("excess_net"),
        (pl.col("long") - pl.col("cost")).alias("long_net"),
    )
    ex_net = daily["excess_net"].to_numpy()
    s_net = _summary(ex_net)
    out = {
        "long_gross_bp": float(daily["long"].mean() * 1e4),
        "universe_bp": float(daily["univ"].mean() * 1e4),
        "excess_gross_bp": float(daily["excess_gross"].mean() * 1e4),
        "cost_bp": float(daily["cost"].mean() * 1e4),
        "excess_net_bp": float(daily["excess_net"].mean() * 1e4),
        "excess_net_ann": float(np.nanmean(ex_net) * TRADING_DAYS),
        "excess_net_sharpe": s_net["ir"] * math.sqrt(TRADING_DAYS),
        "excess_net_t": s_net["t"],
        "excess_net_hit": float(np.mean(ex_net > 0)),
        "excess_net_maxdd": _max_drawdown(ex_net),
        "long_net_ann": float(daily["long_net"].mean() * TRADING_DAYS),
        "top_turnover": float(turn[1:].mean()) if len(turn) > 1 else float("nan"),
        "traded_turnover": float(traded.mean()),
        "avg_top_names": float(daily["n_top"].mean()),
        "avg_universe_names": float(daily["n_univ"].mean()),
        "top_label_missing_frac": float(daily["missing"].mean()),
        "unbuyable_frac": float(1 - panel["buyable"].mean()),
    }
    return out, daily


PORTFOLIOS = {"long_top_vs_universe": long_top_vs_universe}


def evaluate(
    panel: pl.DataFrame,
    rules,
    target: str,
    n_groups: int | None = None,
    allow_locked: bool = False,
) -> dict:
    """Return a flat dict of metrics plus `series` (daily IC and portfolio frames)."""
    p = _check_panel(panel, allow_locked)
    n_groups = n_groups or rules.portfolio.get("n_groups", 10)

    ic = daily_ic(p)
    s_ic = _summary(ic["ic"].to_numpy())
    s_ric = _summary(ic["rank_ic"].to_numpy())

    g = group_returns(p, n_groups)
    gm = g.group_by("group").agg(pl.col("ret").mean()).sort("group")
    group_bp = [float(x * 1e4) for x in gm["ret"].to_list()]
    spread = g.pivot(on="group", index="date", values="ret").sort("date")
    top_bot = (spread[str(n_groups - 1)] - spread["0"]).to_numpy()

    ptype = rules.portfolio["type"]
    if ptype not in PORTFOLIOS:
        raise NotImplementedError(f"portfolio type {ptype!r} is not implemented yet")
    port, port_daily = PORTFOLIOS[ptype](p, rules, target, n_groups)

    metrics = {
        "n_days": int(ic.height),
        "start": str(p["date"].min()),
        "end": str(p["date"].max()),
        "ic": s_ic["mean"],
        "ic_std": s_ic["std"],
        "icir": s_ic["ir"],
        "icir_ann": s_ic["ir"] * math.sqrt(TRADING_DAYS),
        "ic_t": s_ic["t"],
        "rank_ic": s_ric["mean"],
        "rank_icir": s_ric["ir"],
        "rank_ic_t": s_ric["t"],
        "ic_pos_frac": float((ic["ic"] > 0).mean()),
        "group_ret_bp": group_bp,
        "top_minus_bottom_bp": float(np.nanmean(top_bot) * 1e4),
        "monotonicity": float(np.corrcoef(np.arange(n_groups), group_bp)[0, 1]) if len(group_bp) == n_groups else float("nan"),
        **port,
    }
    metrics["series"] = {"ic": ic, "portfolio": port_daily}
    return metrics


def headline(metrics: dict) -> str:
    return (
        f"IC {metrics['ic']:.4f}  RankIC {metrics['rank_ic']:.4f}  ICIR {metrics['icir']:.3f}  "
        f"top-univ {metrics['excess_gross_bp']:.1f}bp gross / {metrics['excess_net_bp']:.1f}bp net  "
        f"net ann {metrics['excess_net_ann']:.1%}  turnover {metrics['top_turnover']:.2f}"
    )
