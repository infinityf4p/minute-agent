"""Read/write standardized parquet datasets.

Layout of one dataset (e.g. data/sim_cn or data/cn):

    security.parquet
    calendar.parquet
    daily/year=YYYY/data.parquet
    minute/year=YYYY/month=MM/data.parquet

All readers go through minute_agent.split: they refuse ranges that reach the
locked test period, never open locked partitions, and mask future information
in the static tables (delist dates, listings after the readable horizon).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from minute_agent import split
from minute_agent.data import schema
from minute_agent.paths import dataset_dir

PARQUET_KW = dict(compression="zstd", compression_level=3, statistics=True)


# ---------------------------------------------------------------------------
# writers (used by ingest / simulators; writing the locked period is allowed)


def write_security(dataset: str, df: pl.DataFrame) -> None:
    root = dataset_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    schema.conform(df, schema.SECURITY_SCHEMA).write_parquet(root / "security.parquet", **PARQUET_KW)


def write_calendar(dataset: str, dates: list[date]) -> None:
    root = dataset_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": sorted(dates)}, schema=schema.CALENDAR_SCHEMA).write_parquet(
        root / "calendar.parquet", **PARQUET_KW
    )


def write_daily_year(dataset: str, year: int, df: pl.DataFrame) -> None:
    p = dataset_dir(dataset) / "daily" / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    schema.conform(df, schema.DAILY_SCHEMA).sort(["date", "symbol"]).write_parquet(
        p / "data.parquet", **PARQUET_KW
    )


def write_minute_month(dataset: str, year: int, month: int, df: pl.DataFrame) -> None:
    p = dataset_dir(dataset) / "minute" / f"year={year}" / f"month={month:02d}"
    p.mkdir(parents=True, exist_ok=True)
    schema.conform(df, schema.MINUTE_SCHEMA).sort(["symbol", "ts"]).write_parquet(
        p / "data.parquet", row_group_size=1_000_000, **PARQUET_KW
    )


# ---------------------------------------------------------------------------
# readers (guarded)


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def minute_months(dataset: str) -> list[tuple[int, int]]:
    """All (year, month) minute partitions on disk that may be opened."""
    root = dataset_dir(dataset) / "minute"
    out = []
    for yd in sorted(root.glob("year=*")):
        for md in sorted(yd.glob("month=*")):
            y, m = int(yd.name[5:]), int(md.name[6:])
            if split.partition_allowed(y, m):
                out.append((y, m))
    return out


def read_minute(
    dataset: str,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    columns: list[str] | None = None,
    max_bidx: int | None = None,
    allow_locked: bool = False,
) -> pl.DataFrame:
    """Minute bars with start <= date <= end. max_bidx drops bars after a cutoff at read time."""
    split.check_range(start, end, allow_locked)
    files: list[Path] = []
    for y, m in _months(start, end):
        if not split.partition_allowed(y, m, allow_locked):
            continue
        f = dataset_dir(dataset) / "minute" / f"year={y}" / f"month={m:02d}" / "data.parquet"
        if f.exists():
            files.append(f)
    cols = columns or list(schema.MINUTE_SCHEMA)
    if not files:
        return pl.DataFrame(schema={c: schema.MINUTE_SCHEMA[c] for c in cols})
    lf = pl.scan_parquet(files).filter(pl.col("date").is_between(start, end))
    if symbols is not None:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if max_bidx is not None:
        lf = lf.filter(pl.col("bidx") <= max_bidx)
    return lf.select(cols).collect()


def read_daily(
    dataset: str,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    columns: list[str] | None = None,
    allow_locked: bool = False,
) -> pl.DataFrame:
    split.check_range(start, end, allow_locked)
    files = []
    for y in range(start.year, end.year + 1):
        if not split.partition_allowed(y, None, allow_locked):
            continue
        f = dataset_dir(dataset) / "daily" / f"year={y}" / "data.parquet"
        if f.exists():
            files.append(f)
    cols = columns or list(schema.DAILY_SCHEMA)
    if not files:
        return pl.DataFrame(schema={c: schema.DAILY_SCHEMA[c] for c in cols})
    lf = pl.scan_parquet(files).filter(pl.col("date").is_between(start, end))
    if symbols is not None:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    return lf.select(cols).sort(["symbol", "date"]).collect()


def read_calendar(dataset: str, start: date, end: date, allow_locked: bool = False) -> list[date]:
    split.check_range(start, end, allow_locked)
    cal = pl.read_parquet(dataset_dir(dataset) / "calendar.parquet")
    return cal.filter(pl.col("date").is_between(start, end))["date"].to_list()


def read_security(dataset: str, allow_locked: bool = False) -> pl.DataFrame:
    """Static security table with future information masked.

    Listings after the readable horizon are dropped, and delist dates after it
    are set to null: knowing in 2023 that a stock will delist in 2025 is a leak.
    """
    horizon = split.max_readable_date(allow_locked)
    sec = pl.read_parquet(dataset_dir(dataset) / "security.parquet")
    return sec.filter(pl.col("list_date") <= horizon).with_columns(
        pl.when(pl.col("delist_date") <= horizon).then(pl.col("delist_date")).alias("delist_date")
    )
