"""Look-ahead check for feature operators.

The information set at (day T, decision bar d) is:
  * minute bars of days < T, and of day T with bidx <= d;
  * daily rows of days < T, plus the pre-open fields of day T
    (pre_close, adj_factor, limits, suspension and ST flags).

check_op computes an operator twice: once on the full data, once on the
information set of each check date T, both starting from the same first day.
If any value for day T differs, the operator used something it could not
have known at the decision time, and LeakageError is raised.

With apply_cutoff=True (production path) the minute data passes through
ops.cutoff first, exactly as in the feature cache. With apply_cutoff=False the
operator gets raw bars, so it must not look past the decision bar on its own;
this is the stricter mode intended for v2 custom feature functions.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from minute_agent.data.schema import DAILY_PREOPEN_FIELDS
from minute_agent.features import ops


class LeakageError(AssertionError):
    pass


def information_set(minute: pl.DataFrame, daily: pl.DataFrame, T: date, d: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    m = minute.filter((pl.col("date") < T) | ((pl.col("date") == T) & (pl.col("bidx") <= d)))
    past = daily.filter(pl.col("date") < T)
    today = daily.filter(pl.col("date") == T)
    hidden = [c for c in daily.columns if c not in DAILY_PREOPEN_FIELDS]
    today = today.with_columns([pl.lit(None, dtype=daily.schema[c]).alias(c) for c in hidden])
    return m, pl.concat([past, today.select(daily.columns)])


def _run(fn, params, minute, daily, d, apply_cutoff):
    m = ops.cutoff(minute, d) if apply_cutoff else minute
    return fn(ops.prepare_minute(m), daily, d, **params)


def _diff_cols(a: pl.DataFrame, b: pl.DataFrame, rtol: float = 1e-9) -> list[str]:
    j = a.join(b, on=ops.KEYS, how="full", suffix="__t", coalesce=True)
    bad = []
    for c in a.columns:
        if c in ops.KEYS:
            continue
        x = j[c].cast(pl.Float64).to_numpy()
        y = j[f"{c}__t"].cast(pl.Float64).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=rtol, atol=1e-12) | both_nan
        if not close.all():
            bad.append(c)
    return bad


def check_op(
    fn,
    params: dict,
    minute: pl.DataFrame,
    daily: pl.DataFrame,
    d: int,
    check_dates: list[date],
    apply_cutoff: bool = True,
) -> None:
    """Raise LeakageError if fn's output on any check date depends on unavailable data."""
    full = _run(fn, params, minute, daily, d, apply_cutoff)
    problems = []
    for T in check_dates:
        m_t, d_t = information_set(minute, daily, T, d)
        trunc = _run(fn, params, m_t, d_t, d, apply_cutoff).filter(pl.col("date") == T)
        ref = full.filter(pl.col("date") == T)
        if trunc.height != ref.height:
            problems.append(f"{T}: {ref.height} rows with full data vs {trunc.height} with information set")
            continue
        bad = _diff_cols(ref, trunc)
        if bad:
            problems.append(f"{T}: columns {bad} change when future data is removed")
    if problems:
        name = getattr(fn, "__name__", str(fn))
        raise LeakageError(f"look-ahead detected in {name}{params}:\n  " + "\n  ".join(problems))


def pick_check_dates(dates: list[date], k: int = 3, seed: int = 0) -> list[date]:
    """k dates spread over the sample, never the last day (it has no future to leak)."""
    ds = sorted(set(dates))[:-1]
    if len(ds) <= k:
        return ds
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ds[len(ds) // 4:], size=k, replace=False).tolist())
