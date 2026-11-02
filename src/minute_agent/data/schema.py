"""Standard schemas for minute bars, daily bars and security info.

Conventions (PLAN.md sections 6 and 10):

* Timestamps are naive exchange-local time (Asia/Shanghai for cn,
  America/New_York for us).
* `ts` is the bar END time. The bar labelled 09:31 covers 09:30:00-09:31:00.
* A-share day = 241 bars, indexed by `bidx`:
    bidx 0         09:30  opening call auction (matched 09:25, printed at 09:30)
    bidx 1..120    09:31 .. 11:30
    bidx 121..240  13:01 .. 15:00
  SZ (all years) and SH (from 2018-08-20) run a closing call auction 14:57-15:00:
  bars 14:58 and 14:59 have zero volume and bar 15:00 is the auction print.
  This matches the vendor files in F:/AshareData (checked 2026-11-02).
* Prices are UNADJUSTED. `adj_factor` is the cumulative back-adjustment factor
  (houfuquan) as of that date. adj_price = price * adj_factor. Its value on day t
  is known at the open of day t and never changes afterwards, so it is
  point-in-time safe (unlike forward-adjusted prices).
* volume is in shares, amount in currency units (CNY / USD).
* A suspended day has a daily row with is_suspended=True, volume=0 and no
  minute bars. Stocks are present from list_date to delist_date (inclusive).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl

MINUTE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,          # e.g. 600268.SH
    "date": pl.Date,
    "ts": pl.Datetime("us"),    # bar end, exchange-local
    "bidx": pl.Int16,           # bar index within the day, see module docstring
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,       # shares
    "amount": pl.Float64,       # currency
}

DAILY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,         # unadjusted
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "pre_close": pl.Float64,    # ex-rights reference price for today's limits
    "volume": pl.Float64,       # shares
    "amount": pl.Float64,
    "turnover": pl.Float64,     # % of float shares
    "adj_factor": pl.Float64,   # cumulative back-adjustment factor
    "up_limit": pl.Float64,     # null = no price limit that day
    "down_limit": pl.Float64,
    "is_suspended": pl.Boolean,
    "is_st": pl.Boolean,
    "float_mv": pl.Float64,     # float market value, currency
    "total_mv": pl.Float64,
}

# Daily fields that are known before the open of that day. Anything else in a
# day-t daily row (open/high/low/close/volume/...) is only known after the close.
DAILY_PREOPEN_FIELDS = (
    "symbol", "date", "pre_close", "adj_factor", "up_limit", "down_limit",
    "is_suspended", "is_st",
)

SECURITY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "exchange": pl.Utf8,        # SH / SZ / BJ (cn), or venue for us
    "board": pl.Utf8,           # main / gem / star / bse
    "list_date": pl.Date,
    "delist_date": pl.Date,     # null = still listed
}

CALENDAR_SCHEMA: dict[str, pl.DataType] = {"date": pl.Date}


# ---------------------------------------------------------------------------
# Session layout


def cn_bar_times() -> list[time]:
    """End times of the 241 A-share bars, bidx 0..240."""
    out = [time(9, 30)]
    t = datetime(2000, 1, 1, 9, 31)
    while t.time() <= time(11, 30):
        out.append(t.time())
        t += timedelta(minutes=1)
    t = datetime(2000, 1, 1, 13, 1)
    while t.time() <= time(15, 0):
        out.append(t.time())
        t += timedelta(minutes=1)
    assert len(out) == 241
    return out


def us_bar_times() -> list[time]:
    """End times of the 390 regular-session US bars, bidx 0..389."""
    out = []
    t = datetime(2000, 1, 1, 9, 31)
    while t.time() <= time(16, 0):
        out.append(t.time())
        t += timedelta(minutes=1)
    assert len(out) == 390
    return out


BAR_TIMES = {"cn": cn_bar_times(), "us": us_bar_times()}


def bar_index_at(market: str, hhmm: str) -> int:
    """bidx of the last bar whose END time is <= hhmm (e.g. '14:50' -> 230 for cn)."""
    h, m = map(int, hhmm.split(":"))
    target = time(h, m)
    times = BAR_TIMES[market]
    idx = [i for i, t in enumerate(times) if t <= target]
    if not idx:
        raise ValueError(f"{hhmm} is before the first bar of {market}")
    return idx[-1]


SH_CLOSE_AUCTION_START = date(2018, 8, 20)


def has_close_auction(symbol: str, d: date) -> bool:
    """A-share closing call auction: SZ always (since 2006), SH from 2018-08-20, BJ yes."""
    if symbol.endswith(".SH"):
        return d >= SH_CLOSE_AUCTION_START
    return True


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select and cast columns to a standard schema; raise if any are missing."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
