"""Ingest the vendor A-share files (F:/AshareData) into the standard schema.

Vendor facts (checked 2026-11-02, see docs/data_quality.md):
* Daily: one CSV per stock-year in 全A日K/<year>.zip. volume in lots, amount in
  thousand CNY, share counts in 10k shares, market values in 10k CNY. No ST flag,
  no limit prices, no rows on suspended days. Delisted stocks seem to be absent.
* Minute: one CSV per stock-year (2016-2025) or stock-day (2026 monthly zips).
  241 bars/day, ts = bar end, volume in lots. `name` is the CURRENT name, so it
  cannot be used for historical ST status. A few stock-days are duplicated.
* The vendor back-adjustment factor jitters ~0.1% day to day, so we rebuild a
  clean cumulative factor from pre_close instead (it changes only on ex-dates).
* BSE (BJ) uses old codes in daily files and new 920xxx codes in minute files;
  BSE is left out of this dataset.

Derived fields (all point-in-time):
* adj_factor_t = prod_{s<=t} close_{s-1} / pre_close_s over traded days.
* is_st_t: inferred from the 5% limit regime using days up to t-1 (see infer_st).
* up/down limits from pre_close, board, inferred ST status and IPO rules.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from minute_agent.data import schema, store
from minute_agent.paths import dataset_dir

RAW = Path("F:/AshareData/股票历史数据")
DAILY_DIR = RAW / "全A日K"
MINUTE_DIR = RAW / "沪深分钟数据" / "Stock_1min_2000-now"

GEM_REFORM = date(2020, 8, 24)
MAIN_REGISTRATION = date(2023, 4, 10)


def board_of(symbol: str) -> str:
    code = symbol[:6]
    if symbol.endswith(".SH"):
        return "star" if code.startswith("68") else "main"
    if symbol.endswith(".SZ"):
        return "gem" if code.startswith("30") else "main"
    return "bse"


# ---------------------------------------------------------------------------
# daily


def _read_daily_zip(path: str) -> pl.DataFrame:
    z = zipfile.ZipFile(path)
    frames = []
    for n in z.namelist():
        if "__MACOSX" in n or not re.search(r"\d{6}\.(SH|SZ)\.csv$", n):
            continue
        df = pl.read_csv(z.read(n), infer_schema_length=0)  # all strings, cast below
        if df.height:
            frames.append(df)
    df = pl.concat(frames, how="diagonal_relaxed")
    num = ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover",
           "float_share", "total_share", "total_mv", "circ_mv"]
    return df.select(
        pl.col("code").alias("symbol"),
        pl.col("datetime").str.to_date().alias("date"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in num],
    )


def round_price(x: pl.Expr) -> pl.Expr:
    """Exchange rounding to 0.01, half up."""
    return ((x * 100 + 0.5 + 1e-9).floor()) / 100


def infer_st(d: pl.DataFrame) -> pl.DataFrame:
    """Point-in-time ST flag for main-board (and pre-reform ChiNext) stocks.

    Evidence on day s: closing exactly at a 5% limit (+5% at the high or -5% at
    the low) without trading beyond +-5.2% that day. Counter-evidence: any trade
    beyond +-5.2% of pre_close. A stock counts as ST on day t if, among days
    before t, the latest counter-evidence is followed by >= 2 evidence days.
    Applies only where the ST limit (5%) differs from the normal one (10%).
    """
    pre = pl.col("pre_close")
    up5 = (pl.col("close") - round_price(pre * 1.05)).abs() < 0.005
    dn5 = (pl.col("close") - round_price(pre * 0.95)).abs() < 0.005
    beyond = (pl.col("high") / pre - 1 > 0.052) | (pl.col("low") / pre - 1 < -0.052)
    hit = ((up5 & (pl.col("close") >= pl.col("high") - 1e-9)) | (dn5 & (pl.col("close") <= pl.col("low") + 1e-9))) & ~beyond
    applicable = (pl.col("board") == "main") | ((pl.col("board") == "gem") & (pl.col("date") < GEM_REFORM))
    d = d.sort("symbol", "date").with_columns(
        hit.fill_null(False).alias("_hit"), beyond.fill_null(False).alias("_beyond")
    ).with_columns(pl.col("_beyond").cast(pl.Int32).cum_sum().over("symbol").alias("_seg"))
    d = d.with_columns(pl.col("_hit").cast(pl.Int32).cum_sum().over("symbol", "_seg").alias("_hits"))
    st_after_close = (pl.col("_hits") >= 2) & applicable
    return d.with_columns(
        st_after_close.shift(1).over("symbol").fill_null(False).alias("is_st")
    ).drop("_hit", "_beyond", "_seg", "_hits")


def limits(d: pl.DataFrame) -> pl.DataFrame:
    """up/down limit prices. Null = no limit (IPO no-limit days)."""
    n = pl.col("trade_idx")   # 0 = listing day
    b, dt = pl.col("board"), pl.col("date")
    pct = (
        pl.when(b == "star").then(0.20)
        .when((b == "gem") & (dt >= GEM_REFORM)).then(0.20)
        .when(pl.col("is_st")).then(0.05)
        .otherwise(0.10)
    )
    no_limit = (
        ((b == "star") & (n < 5))
        | ((b == "gem") & (dt >= GEM_REFORM) & (n < 5))
        | ((b == "main") & (dt >= MAIN_REGISTRATION) & (n < 5))
    )
    ipo_old = (n == 0) & ~no_limit & (b != "star")
    up = pl.when(no_limit).then(None).when(ipo_old).then(round_price(pl.col("pre_close") * 1.44)) \
        .otherwise(round_price(pl.col("pre_close") * (1 + pct)))
    dn = pl.when(no_limit).then(None).when(ipo_old).then(round_price(pl.col("pre_close") * 0.64)) \
        .otherwise(round_price(pl.col("pre_close") * (1 - pct)))
    return d.with_columns(up.alias("up_limit"), dn.alias("down_limit"))


def build_daily(dataset: str = "cn", first_year: int = 2000, last_year: int = 2026,
                write_from: int = 2010, workers: int = 4, verbose: bool = True) -> dict:
    t0 = time.time()
    paths = [str(DAILY_DIR / f"{y}.zip") for y in range(first_year, last_year + 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        raw = pl.concat(list(ex.map(_read_daily_zip, paths)))
    raw = raw.unique(["symbol", "date"], keep="first").sort("symbol", "date")
    if verbose:
        print(f"  daily raw: {raw.height:,} rows, {raw['symbol'].n_unique()} symbols, {time.time() - t0:.0f}s")

    cal = sorted(raw["date"].unique().to_list())
    d = raw.with_columns(
        pl.col("symbol").map_elements(board_of, return_dtype=pl.Utf8).alias("board"),
        pl.int_range(pl.len()).over("symbol").alias("trade_idx"),
        # clean cumulative back-adjustment factor from pre_close
        (pl.col("close").shift(1) / pl.col("pre_close")).over("symbol").fill_null(1.0).alias("_r"),
    )
    bad_ratio = d.filter((pl.col("_r") < 0.05) | (pl.col("_r") > 20)).height
    d = d.with_columns(
        pl.when((pl.col("_r") > 0.05) & (pl.col("_r") < 20)).then(pl.col("_r")).otherwise(1.0)
        .cum_prod().over("symbol").alias("adj_factor")
    ).drop("_r")
    d = infer_st(d)
    d = limits(d)

    # suspended days: calendar days between first and last trade with no row
    cal_df = pl.DataFrame({"date": cal}, schema={"date": pl.Date})
    span = d.group_by("symbol").agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
    grid = span.join(cal_df, how="cross").filter(pl.col("date").is_between(pl.col("first"), pl.col("last")))
    full = grid.select("symbol", "date").join(d, on=["symbol", "date"], how="left").sort("symbol", "date")
    susp = pl.col("close").is_null()
    fill_cols = ["close", "adj_factor", "is_st", "board", "float_share", "total_share"]
    full = full.with_columns(susp.alias("is_suspended")).with_columns(
        [pl.col(c).forward_fill().over("symbol") for c in fill_cols]
    ).with_columns(
        pl.when(pl.col("is_suspended")).then(pl.col("close")).otherwise(pl.col(c)).alias(c)
        for c in ["open", "high", "low", "pre_close"]
    ).with_columns(
        pl.when(pl.col("is_suspended")).then(0.0).otherwise(pl.col(c)).alias(c)
        for c in ["volume", "amount", "turnover"]
    )
    out = full.select(
        "symbol", "date", "open", "high", "low", "close", "pre_close",
        (pl.col("volume") * 100).alias("volume"),
        (pl.col("amount") * 1000).alias("amount"),
        "turnover", "adj_factor", "up_limit", "down_limit", "is_suspended", "is_st",
        (pl.col("close") * pl.col("float_share") * 1e4).alias("float_mv"),
        (pl.col("close") * pl.col("total_share") * 1e4).alias("total_mv"),
    )
    for y in range(write_from, last_year + 1):
        part = out.filter(pl.col("date").dt.year() == y)
        if part.height:
            store.write_daily_year(dataset, y, part)

    end = cal[-1]
    last_idx = {dd: i for i, dd in enumerate(cal)}
    sec = span.with_columns(
        pl.col("symbol").map_elements(board_of, return_dtype=pl.Utf8).alias("board"),
        pl.col("symbol").str.slice(7, 2).alias("exchange"),
        pl.col("first").alias("list_date"),
        pl.when(pl.col("last").map_elements(lambda x: len(cal) - 1 - last_idx[x], return_dtype=pl.Int64) > 10)
        .then(pl.col("last")).otherwise(None).alias("delist_date"),
    )
    store.write_security(dataset, sec)
    store.write_calendar(dataset, cal)
    info = {"daily_rows": out.height, "symbols": sec.height, "calendar_days": len(cal),
            "suspended_rows": int(out["is_suspended"].sum()), "bad_adj_ratio_rows": bad_ratio,
            "seconds": round(time.time() - t0, 1)}
    if verbose:
        print("  daily done:", info)
    return info


# ---------------------------------------------------------------------------
# minute


def _minute_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Vendor minute CSV columns -> standard schema (SH/SZ only)."""
    df = raw.filter(pl.col("code").str.slice(0, 2).is_in(["sh", "sz"])).select(
        (pl.col("code").str.slice(2, 6) + "." + pl.col("code").str.slice(0, 2).str.to_uppercase()).alias("symbol"),
        pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us").alias("ts"),
        pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64),
        (pl.col("volume").cast(pl.Float64) * 100).alias("volume"),
        pl.col("amount").cast(pl.Float64),
    )
    hm = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)
    bidx = pl.when(hm.is_between(570, 690)).then(hm - 570).when(hm.is_between(781, 900)).then(hm - 780 + 120)
    return (
        df.with_columns(pl.col("ts").dt.date().alias("date"), bidx.cast(pl.Int16).alias("bidx"))
        .filter(pl.col("bidx").is_not_null())
        .unique(["symbol", "ts"], keep="first", maintain_order=True)
    )


def _write_parts(dataset: str, df: pl.DataFrame, tag: str) -> None:
    df = df.with_columns(pl.col("date").dt.year().alias("_y"), pl.col("date").dt.month().alias("_m"))
    for (y, m), part in df.partition_by(["_y", "_m"], as_dict=True).items():
        p = dataset_dir(dataset) / "minute" / f"year={y}" / f"month={m:02d}"
        p.mkdir(parents=True, exist_ok=True)
        part = schema.conform(part.drop("_y", "_m"), schema.MINUTE_SCHEMA).sort("symbol", "ts")
        tmp = p / f"part-{tag}.tmp"
        part.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True,
                           row_group_size=1_000_000)
        tmp.replace(p / f"part-{tag}.parquet")


def ingest_minute_zip(dataset: str, zip_path: str, batch_members: int = 300) -> dict:
    t0 = time.time()
    root = dataset_dir(dataset) / "minute"
    zip_tag = Path(zip_path).stem.replace("_1min", "")
    marker = root / f"_done_{zip_tag}"
    if marker.exists():
        return {"zip": zip_tag, "skipped": True}
    z = zipfile.ZipFile(zip_path)
    members = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n
               and re.search(r"/(sh|sz)\d{6}[^/]*\.csv$", n)]
    # stock-day files (2026) are tiny: take more per batch
    if members and "_" not in Path(members[0]).stem:
        batch_members *= 200
    rows = dupes = 0
    for b in range(0, len(members), batch_members):
        frames = []
        for n in members[b:b + batch_members]:
            data = z.read(n)
            if len(data) < 80:
                continue
            frames.append(pl.read_csv(io.BytesIO(data), columns=["datetime", "code", "open", "close", "high",
                                                                 "low", "volume", "amount"],
                                      schema_overrides={"code": pl.Utf8, "datetime": pl.Utf8,
                                                        "volume": pl.Float64, "amount": pl.Float64}))
        if not frames:
            continue
        raw = pl.concat(frames)
        df = _minute_frame(raw)
        dupes += raw.height - df.height
        rows += df.height
        _write_parts(dataset, df, f"{zip_tag}-{b // batch_members:04d}")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{rows} rows, {dupes} dropped\n")
    return {"zip": zip_tag, "rows": rows, "dropped": dupes, "members": len(members),
            "seconds": round(time.time() - t0, 1)}


def minute_zips(first_year: int = 2016) -> list[str]:
    out = []
    for p in sorted(MINUTE_DIR.glob("*.zip")):
        m = re.match(r"(\d{4})", p.name)
        if m and int(m.group(1)) >= first_year:
            out.append(str(p))
    return out


def build_minute(dataset: str = "cn", first_year: int = 2016, workers: int = 3, verbose: bool = True) -> list[dict]:
    import multiprocessing as mp

    from minute_agent._mp import init_worker

    zips = minute_zips(first_year)
    # biggest first so the pool stays busy
    zips.sort(key=lambda p: -Path(p).stat().st_size)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(max(1, 20 // workers),)) as ex:
        futs = [ex.submit(ingest_minute_zip, dataset, z) for z in zips]
        for f in futs:
            r = f.result()
            results.append(r)
            if verbose:
                print("  minute", r, flush=True)
    return results
