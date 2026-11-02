"""Fixed rule for the T2 base position (not searchable by the agent).

On the first trading day of each month, take the N most liquid stocks by
20-day average turnover value (up to the previous day) among stocks that are
not ST, not suspended and listed for at least 60 trading days. Hold them
equal-weight for the month; the base value per stock is capital / N.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from minute_agent.data import store


def base_positions(dataset: str, start: date, end: date, n: int = 300, capital: float = 1e8) -> pl.DataFrame:
    cal = store.read_calendar(dataset, start - timedelta(days=400), end)
    daily = store.read_daily(dataset, start - timedelta(days=60), end,
                             columns=["symbol", "date", "amount", "is_st", "is_suspended"])
    sec = store.read_security(dataset).select("symbol", "list_date")
    adv = daily.sort("symbol", "date").with_columns(
        pl.col("amount").rolling_mean(20, min_samples=10).shift(1).over("symbol").alias("adv20"))
    days = [d for d in cal if start <= d <= end]
    firsts = sorted({min(d for d in days if (d.year, d.month) == ym) for ym in {(d.year, d.month) for d in days}})
    cal_arr = np.array(cal, dtype="datetime64[D]")
    rows = []
    for i, d0 in enumerate(firsts):
        d1 = firsts[i + 1] if i + 1 < len(firsts) else end + timedelta(days=1)
        k = int(np.searchsorted(cal_arr, np.datetime64(d0)))
        min_list = cal[max(0, k - 60)]
        snap = adv.filter(pl.col("date") == d0).join(sec, on="symbol").filter(
            ~pl.col("is_st") & ~pl.col("is_suspended") & (pl.col("list_date") <= min_list)
            & pl.col("adv20").is_not_null()
        ).sort("adv20", descending=True).head(n)
        month_days = [d for d in days if d0 <= d < d1]
        for s in snap["symbol"].to_list():
            rows.extend((s, d) for d in month_days)
    base = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "date": pl.Date}, orient="row")
    per = capital / n
    # a base stock that is suspended on a day can't be traded; keep the row, the engine sees no prices
    return base.with_columns(pl.lit(per).alias("base_value"))
