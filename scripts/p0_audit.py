"""P0: where does the baseline's validation IC of ~0.32 come from?

Uses saved predictions of the M1 baselines (validation 2022-2023) and splits
the overnight label into pieces, re-anchors it at a tradable price, removes
untradable rows, and winsorizes. Also runs a look-ahead canary on real data.

    conda run -n minute-agent python scripts/p0_audit.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minute_agent.data import store  # noqa: E402
from minute_agent.features import ops  # noqa: E402
from minute_agent.features.leakage import LeakageError, check_op, pick_check_dates  # noqa: E402
from minute_agent.paths import runs_root  # noqa: E402

DS = "cn"
START, END = date(2022, 1, 1), date(2023, 12, 31)


def day_ic(df: pl.DataFrame, y: str) -> dict:
    d = df.filter(pl.col(y).is_not_null() & pl.col(y).is_finite())
    g = d.with_columns(pl.col("score").rank().over("date").alias("_rs"), pl.col(y).rank().over("date").alias("_ry")) \
        .group_by("date").agg(pl.corr("score", y).alias("ic"), pl.corr("_rs", "_ry").alias("ric"), pl.len().alias("n")) \
        .filter(pl.col("n") >= 50)
    ic, ric = g["ic"].to_numpy(), g["ric"].to_numpy()
    return {"ic": float(np.nanmean(ic)), "rank_ic": float(np.nanmean(ric)),
            "icir": float(np.nanmean(ic) / np.nanstd(ic)), "days": g.height, "rows": d.height}


def deciles(df: pl.DataFrame, y: str) -> list[float]:
    d = df.filter(pl.col(y).is_not_null() & pl.col(y).is_finite()).with_columns(
        ((pl.col("score").rank(method="ordinal").over("date") - 1) * 10 / pl.len().over("date")).floor().alias("g"))
    m = d.group_by("date", "g").agg(pl.col(y).mean()).group_by("g").agg(pl.col(y).mean()).sort("g")
    return [round(float(v) * 1e4, 1) for v in m[y].to_list()]


def build_targets() -> pl.DataFrame:
    cal = store.read_calendar(DS, START, END)
    nxt = pl.DataFrame({"date": cal[:-1], "nd": cal[1:]}, schema={"date": pl.Date, "nd": pl.Date})
    d = store.read_daily(DS, START - timedelta(days=10), END)
    base = d.select("symbol", "date", "open", "close", "adj_factor", "up_limit", "down_limit", "is_st", "is_suspended")
    n = base.select(pl.col("symbol"), pl.col("date").alias("nd"), pl.col("open").alias("n_open"),
                    pl.col("close").alias("n_close"), pl.col("adj_factor").alias("n_adj"),
                    pl.col("is_suspended").alias("n_susp"), pl.col("up_limit").alias("n_up"), pl.col("down_limit").alias("n_dn"))
    parts = []
    for y in (2022, 2023):
        for mo in range(1, 13):
            m0 = date(y, mo, 1)
            m1 = (date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)) - timedelta(days=1)
            m = store.read_minute(DS, m0, m1, columns=["symbol", "date", "bidx", "close", "volume", "amount"])
            parts.append(m.filter(pl.col("bidx") >= 230).group_by("symbol", "date").agg(
                pl.col("close").filter(pl.col("bidx") == 230).last().alias("p1450"),
                (pl.col("amount").filter(pl.col("bidx").is_between(231, 235)).sum()
                 / pl.col("volume").filter(pl.col("bidx").is_between(231, 235)).sum()).alias("vwap1451_55"),
                pl.col("close").filter(pl.col("bidx") == 236).last().alias("p1456"),
            ))
    late = pl.concat(parts)
    t = base.filter(pl.col("date").is_between(START, END)).join(nxt, on="date").join(n, on=["symbol", "nd"], how="left") \
        .join(late, on=["symbol", "date"], how="left")
    adj_ratio = pl.col("n_adj") / pl.col("adj_factor")
    ok_next = ~pl.col("n_susp").fill_null(True)
    return t.with_columns(
        (pl.col("close") / pl.col("p1450") - 1).alias("r_1450_close"),
        pl.when(ok_next).then(pl.col("n_open") * adj_ratio / pl.col("close") - 1).alias("r_close_nopen"),
        pl.when(ok_next).then(pl.col("n_close") / pl.col("n_open") - 1).alias("r_nopen_nclose"),
        pl.when(ok_next).then(pl.col("n_open") * adj_ratio / pl.col("vwap1451_55") - 1).alias("r_vwap_nopen"),
        pl.when(ok_next).then(pl.col("n_open") * adj_ratio / pl.col("p1450") - 1).alias("r_1450_nopen"),
        ((pl.col("p1450") >= pl.col("up_limit") - 1e-6) | (pl.col("p1450") <= pl.col("down_limit") + 1e-6)).fill_null(False).alias("lim_1450"),
        ((pl.col("close") >= pl.col("up_limit") - 1e-6) | (pl.col("close") <= pl.col("down_limit") + 1e-6)).fill_null(False).alias("lim_close"),
        ((pl.col("n_open") >= pl.col("n_up") - 1e-6) | (pl.col("n_open") <= pl.col("n_dn") + 1e-6)).fill_null(False).alias("lim_nopen"),
    )


def canary() -> dict:
    def cheat_next_open(m, daily, d):
        first = m.filter(pl.col("bidx") == 0).select("symbol", "date", pl.col("close").alias("o")).sort("symbol", "date") \
            .with_columns(pl.col("o").shift(-1).over("symbol").alias("no"))
        last = m.group_by("symbol", "date").agg(pl.col("close").last().alias("c"))
        return last.join(first, on=["symbol", "date"]).select("symbol", "date", (pl.col("no") / pl.col("c")).log().alias("peek"))
    s, e = date(2022, 6, 1), date(2022, 6, 30)
    minute = store.read_minute(DS, s, e)
    syms = sorted(minute["symbol"].unique().to_list())[::25]
    minute = minute.filter(pl.col("symbol").is_in(syms))
    daily = store.read_daily(DS, s - timedelta(days=120), e, symbols=syms)
    dates = pick_check_dates(minute["date"].unique().to_list(), k=3)
    out = {"symbols": len(syms)}
    try:
        check_op(cheat_next_open, {}, minute, daily, 230, dates)
        out["cheat_caught"] = False
    except LeakageError:
        out["cheat_caught"] = True
    try:
        check_op(ops.REGISTRY["ret"][0], {"window_min": [5, 30]}, minute, daily, 230, dates)
        out["honest_passed"] = True
    except LeakageError:
        out["honest_passed"] = False
    return out


def main():
    run_ids = sys.argv[1:]
    t = build_targets()
    segs = ["label", "r_1450_close", "r_close_nopen", "r_nopen_nclose", "r_1450_nopen", "r_vwap_nopen"]
    result = {"canary": canary(), "runs": {}}
    for rid in run_ids:
        cfg = json.loads((runs_root() / rid / "config.json").read_text())
        p = pl.read_parquet(runs_root() / rid / "predictions.parquet")
        df = p.join(t, on=["symbol", "date"], how="left")
        tradable = ~pl.col("lim_1450") & ~pl.col("lim_close") & ~pl.col("is_st").fill_null(False)
        r = {"name": cfg.get("name"), "rows": df.height,
             "share_limit_at_1450": float(df["lim_1450"].mean()), "share_limit_at_close": float(df["lim_close"].mean()),
             "share_limit_next_open": float(df["lim_nopen"].mean()),
             "segments_all": {s: day_ic(df, s) for s in segs},
             "segments_tradable": {s: day_ic(df.filter(tradable), s) for s in segs},
             "deciles_bp": {s: deciles(df.filter(tradable), s) for s in ["label", "r_vwap_nopen", "r_1450_close", "r_nopen_nclose"]}}
        # winsorize label 1%/99% per day
        w = df.with_columns(pl.col("label").clip(pl.col("label").quantile(0.01).over("date"),
                                                 pl.col("label").quantile(0.99).over("date")).alias("label_w"))
        r["label_winsorized"] = day_ic(w, "label_w")
        # contribution of the most extreme 1% of labels to Pearson IC
        r["label_without_top1pct_abs"] = day_ic(df.filter(pl.col("label").abs() < pl.col("label").abs().quantile(0.99).over("date")), "label")
        result["runs"][rid] = r
        print(json.dumps(r, indent=1), flush=True)
    out = runs_root() / "p0_audit.json"
    out.write_text(json.dumps(result, indent=1))
    print("canary:", result["canary"])
    print("saved", out)


if __name__ == "__main__":
    main()
