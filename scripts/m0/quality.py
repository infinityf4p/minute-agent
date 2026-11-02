"""M0 data-quality report for the ingested cn dataset -> docs/data_quality.md (+ JSON).

Locked-period partitions (2024+) are read directly from parquet here, for
mechanical checks only (counts, duplicates, minute/daily consistency). No
returns or signals of the locked period are computed.

    conda run -n minute-agent python scripts/m0/quality.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from minute_agent.data.ingest_cn import DAILY_DIR, MINUTE_DIR, RAW  # noqa: E402
from minute_agent.paths import dataset_dir  # noqa: E402

DS = "cn"
ROOT = dataset_dir(DS)


def month_dirs():
    for yd in sorted((ROOT / "minute").glob("year=*")):
        for md in sorted(yd.glob("month=*")):
            yield int(yd.name[5:]), int(md.name[6:]), md


def minute_month_stats(y, m, md):
    lf = pl.scan_parquet(sorted(md.glob("*.parquet")))
    per_day = lf.group_by("symbol", "date").agg(
        pl.len().alias("bars"),
        pl.col("ts").n_unique().alias("uniq"),
        pl.col("open").filter(pl.col("bidx") == 0).first().alias("m_open"),
        pl.col("close").filter(pl.col("bidx") == 240).last().alias("m_close"),
        pl.col("volume").sum().alias("m_vol"),
        pl.col("amount").sum().alias("m_amt"),
        pl.col("volume").filter(pl.col("bidx").is_in([238, 239])).sum().alias("v_5859"),
    ).collect()
    return per_day


def main():
    out = {}
    daily = pl.scan_parquet(sorted((ROOT / "daily").glob("year=*/*.parquet"))).filter(
        pl.col("date") >= date(2016, 1, 1)).collect()
    sec = pl.read_parquet(ROOT / "security.parquet")
    cal = pl.read_parquet(ROOT / "calendar.parquet")["date"]
    traded = daily.filter(~pl.col("is_suspended"))

    # --- minute vs daily, month by month
    rows = []
    for y, m, md in month_dirs():
        s = minute_month_stats(y, m, md)
        s = s.with_columns(pl.lit(y).alias("year"))
        rows.append(s)
        print(f"  {y}-{m:02d}: {s.height:,} stock-days", flush=True)
    md_all = pl.concat(rows)
    j = traded.select("symbol", "date", "open", "close", "volume", "amount").join(
        md_all, on=["symbol", "date"], how="full", coalesce=True)
    j = j.with_columns(pl.col("date").dt.year().alias("year"))
    per_year = j.group_by("year").agg(
        pl.col("open").is_not_null().sum().alias("daily_stock_days"),
        pl.col("bars").is_not_null().sum().alias("minute_stock_days"),
        (pl.col("open").is_not_null() & pl.col("bars").is_null()).sum().alias("daily_without_minute"),
        (pl.col("open").is_null() & pl.col("bars").is_not_null()).sum().alias("minute_without_daily"),
        (pl.col("bars") != 241).sum().alias("days_not_241_bars"),
        (pl.col("bars") != pl.col("uniq")).sum().alias("days_with_dup_ts"),
        ((pl.col("m_open") - pl.col("open")).abs() > 0.0051).sum().alias("open_mismatch"),
        ((pl.col("m_close") - pl.col("close")).abs() > 0.0051).sum().alias("close_mismatch"),
        ((pl.col("m_vol") / pl.col("volume") - 1).abs() > 0.01).sum().alias("volume_mismatch_1pct"),
        pl.col("symbol").filter(pl.col("bars").is_not_null()).n_unique().alias("minute_symbols"),
        pl.col("symbol").filter(pl.col("open").is_not_null()).n_unique().alias("daily_symbols"),
    ).sort("year")
    out["minute_vs_daily"] = per_year.to_dicts()

    # --- closing auction pattern: SH before 2018-08-20 should trade at 14:58/14:59
    ca = md_all.with_columns(pl.col("symbol").str.ends_with(".SH").alias("sh")).group_by(
        "year", "sh").agg((pl.col("v_5859") > 0).mean().alias("frac_days_trading_1458_1459")).sort("year", "sh")
    out["closing_auction"] = ca.to_dicts()

    # --- daily: suspension, limits, ST, adj factor
    d = daily.with_columns(pl.col("date").dt.year().alias("year"))
    t = d.filter(~pl.col("is_suspended"))
    out["daily_by_year"] = t.group_by("year").agg(
        pl.col("symbol").n_unique().alias("symbols_traded"),
        pl.len().alias("stock_days"),
        (pl.col("close") >= pl.col("up_limit") - 1e-6).mean().alias("frac_close_at_up"),
        (pl.col("close") <= pl.col("down_limit") + 1e-6).mean().alias("frac_close_at_down"),
        (pl.col("high") > pl.col("up_limit") + 0.0051).sum().alias("high_above_up_limit"),
        (pl.col("low") < pl.col("down_limit") - 0.0051).sum().alias("low_below_down_limit"),
        pl.col("is_st").mean().alias("frac_st_inferred"),
        (pl.col("adj_factor") != pl.col("adj_factor").shift(1).over("symbol")).mean().alias("frac_ex_days"),
    ).join(d.group_by("year").agg(pl.col("is_suspended").mean().alias("frac_suspended")), on="year").sort("year").to_dicts()

    # --- survivorship: how many stocks stop trading before the end of the data?
    end = cal.max()
    last = daily.group_by("symbol").agg(pl.col("date").max().alias("last"))
    out["survivorship"] = {
        "symbols_total": sec.height,
        "symbols_with_last_trade_before_2026_06": int(last.filter(pl.col("last") < date(2026, 6, 1)).height),
        "delist_date_set": int(sec["delist_date"].is_not_null().sum()),
        "known_delisted_examples_present": {c: bool(sec.filter(pl.col("symbol") == c).height)
                                            for c in ["000979.SZ", "000939.SZ", "600145.SH", "600087.SH"]},
    }

    # --- vendor adj factor noise vs ours
    z = zipfile.ZipFile(RAW / "复权因子" / "复权因子_后复权.zip")
    noise, agree = [], []
    rng = np.random.default_rng(0)
    syms = rng.choice(sorted(t["symbol"].unique().to_list()), 200, replace=False)
    for s in syms:
        try:
            v = pl.read_csv(z.read(f"{s}.csv"), new_columns=["symbol", "date", "f"]).with_columns(
                pl.col("date").str.to_date())
        except KeyError:
            continue
        ours = d.filter(pl.col("symbol") == s).select("date", "adj_factor", "is_suspended")
        x = ours.join(v.select("date", "f"), on="date").sort("date").filter(~pl.col("is_suspended")).with_columns(
            (pl.col("adj_factor") / pl.col("adj_factor").shift(1)).alias("ro"), (pl.col("f") / pl.col("f").shift(1)).alias("rv"))
        x = x.drop_nulls()
        nonex = x.filter((pl.col("ro") - 1).abs() < 1e-12)
        ex = x.filter((pl.col("ro") - 1).abs() >= 1e-12)
        if nonex.height:
            noise.append(float((nonex["rv"] - 1).abs().median()))
        if ex.height:
            agree.append(float(((ex["rv"] / ex["ro"]) - 1).abs().median()))
    out["adj_factor"] = {"vendor_median_abs_daily_change_on_non_ex_days": float(np.median(noise)) if noise else None,
                         "median_abs_rel_diff_on_ex_days_vendor_vs_ours": float(np.median(agree)) if agree else None,
                         "stocks_checked": len(syms)}

    # --- ST inference vs current names (names in the vendor files are current, i.e. as of late 2026)
    zname = zipfile.ZipFile(MINUTE_DIR / "2026-10_1min.zip")
    names = {}
    for n in zname.namelist():
        mm = re.search(r"/(sh|sz)(\d{6})\.csv$", n)
        if not mm:
            continue
        sym = f"{mm.group(2)}.{mm.group(1).upper()}"
        if sym in names:
            continue
        with zname.open(n) as f:
            f.readline()
            line = f.readline().decode("utf-8", "ignore")
        if line:
            names[sym] = line.split(",")[2]
    lastday = d.filter(pl.col("date") == end).select("symbol", "is_st")
    nm = pl.DataFrame({"symbol": list(names), "name": list(names.values())}).with_columns(
        pl.col("name").str.contains("ST").alias("st_name"))
    k = lastday.join(nm, on="symbol").join(sec.select("symbol", "board"), on="symbol")
    km = k.filter(pl.col("board") == "main")
    tp = km.filter(pl.col("is_st") & pl.col("st_name")).height
    out["st_inference_vs_current_names_main_board"] = {
        "date": str(end), "st_by_name": int(km["st_name"].sum()), "st_inferred": int(km["is_st"].sum()),
        "true_positive": tp,
        "precision": tp / max(int(km["is_st"].sum()), 1), "recall": tp / max(int(km["st_name"].sum()), 1),
    }

    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "runs").mkdir(exist_ok=True)
    (REPO / "runs" / "data_quality.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
