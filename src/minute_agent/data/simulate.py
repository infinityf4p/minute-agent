"""Synthetic A-share style minute + daily data, in the standard schema.

Covers the cases the real pipeline must handle: 241 bars/day with opening and
closing call auctions (no SH closing auction before 2018-08-20), price limits by
board / ST status / date, IPOs (with no-limit first days), suspensions,
delistings, ST spells, cash dividends and bonus shares (unadjusted prices +
cumulative back-adjustment factor).

Planted signals, so the pipeline has something to find:
  a_t  afternoon reversal: price drifts against a_t between 13:30 and 14:50,
       overnight return of t+1 loads on a_t.
  b_t  late volume surge 14:21-14:50 predicts a positive overnight return.
  c_t  a move that happens only AFTER 14:50; it also predicts the overnight
       return. A feature that peeks past the decision time looks great on c_t,
       which is what the leakage tests rely on.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from minute_agent.data import schema, store

LUNAR_NEW_YEAR = {
    2015: date(2015, 2, 19), 2016: date(2016, 2, 8), 2017: date(2017, 1, 28),
    2018: date(2018, 2, 16), 2019: date(2019, 2, 5), 2020: date(2020, 1, 25),
    2021: date(2021, 2, 12), 2022: date(2022, 2, 1), 2023: date(2023, 1, 22),
    2024: date(2024, 2, 10), 2025: date(2025, 1, 29), 2026: date(2026, 2, 17),
}

GEM_20PCT_START = date(2020, 8, 24)
STAR_START = date(2019, 7, 22)
BSE_START = date(2021, 11, 15)


def sim_calendar(start: date, end: date) -> list[date]:
    """Weekdays minus approximate Chinese holidays (good enough for simulation)."""
    hol: set[date] = set()
    for y in range(start.year, end.year + 1):
        hol |= {date(y, 1, 1), date(y, 5, 1), date(y, 5, 2), date(y, 5, 3), date(y, 4, 4), date(y, 4, 5)}
        hol |= {date(y, 10, d) for d in range(1, 8)}
        if y in LUNAR_NEW_YEAR:
            lny = LUNAR_NEW_YEAR[y]
            hol |= {lny + timedelta(days=k) for k in range(-1, 6)}
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


@dataclass
class SimConfig:
    n_stocks: int = 300
    start: date = date(2016, 1, 1)
    end: date = date(2026, 10, 31)
    seed: int = 7
    frac_ipo: float = 0.30        # share of stocks that list inside the sample
    frac_delist: float = 0.06
    frac_st: float = 0.06         # random ST spells (delisting stocks are ST anyway)
    p_suspend: float = 0.0015     # daily probability that a suspension starts
    signal: dict = field(default_factory=lambda: {"a": 0.06, "b": 0.05, "c": 0.08})


BOARDS = ["main_sh", "main_sz", "gem", "star", "bse"]
BOARD_P = [0.36, 0.26, 0.22, 0.11, 0.05]


def _limit_pct(board: str, is_st: bool, d: date) -> float:
    if board == "bse":
        return 0.30
    if board == "star":
        return 0.20
    if board == "gem":
        return 0.20 if d >= GEM_20PCT_START else (0.05 if is_st else 0.10)
    return 0.05 if is_st else 0.10


def _make_securities(cfg: SimConfig, cal: list[date], rng: np.random.Generator) -> pl.DataFrame:
    n = cfg.n_stocks
    boards = rng.choice(len(BOARDS), size=n, p=BOARD_P)
    counters = {b: 0 for b in BOARDS}
    prefix = {"main_sh": (600000, "SH"), "main_sz": (1, "SZ"), "gem": (300001, "SZ"),
              "star": (688001, "SH"), "bse": (830001, "BJ")}
    rows = []
    cal_arr = np.array(cal, dtype="datetime64[D]")
    for i in range(n):
        b = BOARDS[boards[i]]
        if {"star": STAR_START, "bse": BSE_START}.get(b, cal[0]) > cal[-31]:
            b = "main_sh"  # board did not exist yet in this sample window
        base, exch = prefix[b]
        code = f"{base + counters[b]:06d}.{exch}"
        counters[b] += 1
        earliest = {"star": STAR_START, "bse": BSE_START}.get(b)
        if earliest is not None or rng.random() < cfg.frac_ipo:
            lo = max(cal[0], earliest or cal[0])
            candidates = cal_arr[cal_arr >= np.datetime64(lo)]
            list_date = candidates[rng.integers(0, len(candidates) - 30)].astype(date)
        else:
            list_date = cfg.start - timedelta(days=int(rng.integers(200, 6000)))
        rows.append({"symbol": code, "exchange": exch,
                     "board": "main" if b.startswith("main") else b,
                     "list_date": list_date, "delist_date": None})
    sec = pl.DataFrame(rows, schema=schema.SECURITY_SCHEMA)

    # delistings: pick among stocks listed well before the end
    eligible = np.where(np.array(sec["list_date"].to_list()) < cal[-1] - timedelta(days=900))[0]
    n_del = int(round(cfg.frac_delist * n))
    chosen = rng.choice(eligible, size=min(n_del, len(eligible)), replace=False)
    delist = [None] * n
    for i in chosen:
        lo = max(sec["list_date"][int(i)] + timedelta(days=600), cal[len(cal) // 5])
        cands = cal_arr[cal_arr >= np.datetime64(lo)]
        if len(cands) < 2:
            continue
        delist[int(i)] = cands[rng.integers(0, len(cands) - 1)].astype(date)
    return sec.with_columns(pl.Series("delist_date", delist, dtype=pl.Date))


def _st_spells(cfg: SimConfig, sec: pl.DataFrame, cal: list[date], rng) -> list[list[tuple[date, date]]]:
    idx = {d: k for k, d in enumerate(cal)}
    spells: list[list[tuple[date, date]]] = [[] for _ in range(sec.height)]
    for i, (ld, dd) in enumerate(zip(sec["list_date"], sec["delist_date"])):
        if dd is not None:  # ST for ~120 trading days before delisting
            k = idx.get(dd, len(cal) - 1)
            spells[i].append((cal[max(0, k - 120)], dd))
        elif rng.random() < cfg.frac_st:
            k0 = int(rng.integers(0, max(1, len(cal) - 130)))
            k1 = min(len(cal) - 1, k0 + int(rng.integers(120, 500)))
            spells[i].append((cal[k0], cal[k1]))
    return spells


def _corporate_actions(sec: pl.DataFrame, cal: list[date], rng) -> dict[tuple[int, date], tuple[float, float]]:
    """(stock, ex_date) -> (cash dividend as fraction of price, share ratio)."""
    by_year: dict[int, list[date]] = {}
    for d in cal:
        if d.month in (6, 7):
            by_year.setdefault(d.year, []).append(d)
    out = {}
    for i in range(sec.height):
        for y, days in by_year.items():
            if rng.random() < 0.65:
                d = days[int(rng.integers(0, len(days)))]
                ratio = float(rng.choice([1.2, 1.3, 1.5, 2.0])) if rng.random() < 0.08 else 1.0
                out[(i, d)] = (float(rng.uniform(0.003, 0.03)), ratio)
    return out


def _volume_profile() -> np.ndarray:
    k = np.arange(241, dtype=float)
    w = 1 + 2.5 * np.exp(-k / 15) + 1.5 * np.exp(-(240 - k) / 15)
    w[0] = 4.0      # opening auction
    w[240] = 5.0    # closing auction
    return w / w.sum()


def generate(dataset: str, cfg: SimConfig | None = None, verbose: bool = True) -> dict:
    """Generate a full dataset and write it with minute_agent.data.store."""
    cfg = cfg or SimConfig()
    t0 = _time.time()
    rng = np.random.default_rng(cfg.seed)
    cal = sim_calendar(cfg.start, cfg.end)
    sec = _make_securities(cfg, cal, rng)
    spells = _st_spells(cfg, sec, cal, rng)
    actions = _corporate_actions(sec, cal, rng)
    store.write_security(dataset, sec)
    store.write_calendar(dataset, cal)

    n = sec.height
    syms = np.array(sec["symbol"].to_list())
    boards = sec["board"].to_list()
    list_d = sec["list_date"].to_list()
    delist_d = sec["delist_date"].to_list()
    is_sh = np.array([s.endswith(".SH") for s in syms])

    # per-stock parameters
    sigma = rng.uniform(0.018, 0.035, n)
    beta = rng.uniform(0.7, 1.3, n)
    float_sh = np.exp(rng.uniform(np.log(1e8), np.log(5e9), n))
    total_sh = float_sh * rng.uniform(1.0, 1.6, n)
    turn_rate = rng.uniform(0.005, 0.04, n)
    issue_px = np.round(np.exp(rng.normal(np.log(12), 0.6, n)).clip(3, 80), 2)

    # state
    prev_close = issue_px.copy()
    adj = np.ones(n)
    sig_a = np.zeros(n)
    sig_b = np.zeros(n)
    sig_c = np.zeros(n)
    susp_left = np.zeros(n, dtype=int)
    days_listed = np.array([1000 if ld < cal[0] else 0 for ld in list_d], dtype=int)
    gap_drift = np.zeros(n)

    prof = _volume_profile()
    ushape = np.sqrt(np.concatenate([[1.0], 1 + 1.5 * np.exp(-np.arange(240) / 20) + 0.8 * np.exp(-(239 - np.arange(240)) / 20)]))
    bar_us = np.array(
        [(t.hour * 3600 + t.minute * 60) * 1_000_000 for t in schema.cn_bar_times()], dtype=np.int64
    )
    bidx = np.arange(241, dtype=np.int16)
    k_a, k_b, k_c = cfg.signal["a"], cfg.signal["b"], cfg.signal["c"]

    minute_buf: list[pl.DataFrame] = []
    daily_buf: list[dict] = []
    cur_month = (cal[0].year, cal[0].month)
    cur_year = cal[0].year
    n_minute_rows = 0

    def flush_month():
        nonlocal minute_buf, n_minute_rows
        if minute_buf:
            df = pl.concat(minute_buf)
            n_minute_rows += df.height
            store.write_minute_month(dataset, cur_month[0], cur_month[1], df)
        minute_buf = []

    def flush_year():
        nonlocal daily_buf
        if daily_buf:
            store.write_daily_year(dataset, cur_year, pl.DataFrame(daily_buf, schema=schema.DAILY_SCHEMA))
        daily_buf = []

    for d in cal:
        if (d.year, d.month) != cur_month:
            flush_month()
            cur_month = (d.year, d.month)
        if d.year != cur_year:
            flush_year()
            cur_year = d.year
            if verbose:
                print(f"  sim {cur_year} ... {_time.time() - t0:.0f}s", flush=True)

        listed = np.array([(ld <= d) and (dd is None or d <= dd) for ld, dd in zip(list_d, delist_d)])
        if not listed.any():
            continue
        is_st = np.array([any(a <= d <= b for a, b in sp) for sp in spells])
        first_day = np.array([ld == d for ld in list_d])

        # pre-close and corporate actions
        pre_close = prev_close.copy()
        for i in np.where(listed)[0]:
            act = actions.get((int(i), d))
            if act is not None and not first_day[i]:
                div, ratio = act
                new_pc = round((prev_close[i] * (1 - div)) / ratio, 2)
                adj[i] *= prev_close[i] / new_pc
                pre_close[i] = new_pc
        pre_close[first_day] = issue_px[first_day]

        # limits
        up = np.full(n, np.nan)
        dn = np.full(n, np.nan)
        for i in np.where(listed)[0]:
            b = boards[i]
            if first_day[i] and b == "main":
                up[i], dn[i] = round(pre_close[i] * 1.44, 2), round(pre_close[i] * 0.64, 2)
            elif days_listed[i] < 5 and (b in ("star", "bse") or (b == "gem" and d >= GEM_20PCT_START)):
                continue  # no price limit in the first 5 days
            else:
                p = _limit_pct(b, bool(is_st[i]), d)
                up[i], dn[i] = round(pre_close[i] * (1 + p), 2), round(pre_close[i] * (1 - p), 2)
        up_c = np.where(np.isnan(up), np.inf, up)
        dn_c = np.where(np.isnan(dn), 0.0, dn)

        # suspensions
        start_susp = listed & (susp_left == 0) & ~first_day & (rng.random(n) < cfg.p_suspend)
        susp_left[start_susp] = np.where(rng.random(start_susp.sum()) < 0.08,
                                         rng.integers(20, 120, start_susp.sum()),
                                         rng.integers(1, 8, start_susp.sum()))
        suspended = listed & (susp_left > 0)
        active = listed & ~suspended

        mkt_on = rng.normal(-0.0004, 0.004)
        mkt_min = rng.normal(0.00015 / 240, 0.011 / np.sqrt(240), 240)

        A = np.where(active)[0]
        m = len(A)
        if m:
            sA = sigma[A] * np.exp(rng.normal(0, 0.25, m))   # day-specific vol
            delisting = np.array([delist_d[i] is not None for i in A]) & is_st[A]
            r_on = (beta[A] * mkt_on
                    + (k_a * sig_a[A] + k_b * sig_b[A] + k_c * sig_c[A]) * sigma[A]
                    + rng.normal(0, 0.35, m) * sigma[A]
                    + gap_drift[A])
            r_on[first_day[A]] = rng.normal(0.25, 0.15, first_day[A].sum())
            p0 = np.round(np.clip(pre_close[A] * np.exp(r_on), dn_c[A], up_c[A]), 2)

            a_t, b_t, c_t = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)
            eps = rng.normal(size=(m, 240)) * (sA[:, None] / np.sqrt(240)) * ushape[None, 1:]
            lr = beta[A, None] * mkt_min[None, :] + eps
            lr += np.where(delisting, -0.004 / 240, 0.0)[:, None]
            lr[:, 150:230] += (-0.25 * a_t * sA / 80)[:, None]        # 13:31-14:50
            lr[:, 230:236] += (0.20 * c_t * sA / 6)[:, None]          # 14:51-14:56, after decision
            path = p0[:, None] * np.exp(np.cumsum(lr, axis=1))
            path = np.round(np.clip(path, dn_c[A, None], up_c[A, None]), 2)
            closes = np.concatenate([p0[:, None], path], axis=1)          # (m, 241)

            auction = np.array([schema.has_close_auction(syms[i], d) for i in A])
            last_cont = closes[:, 237].copy()
            fin = np.round(np.clip(last_cont * np.exp(rng.normal(0, 0.001, m)), dn_c[A], up_c[A]), 2)
            closes[auction, 238] = last_cont[auction]
            closes[auction, 239] = last_cont[auction]
            closes[auction, 240] = fin[auction]

            opens = np.concatenate([p0[:, None], closes[:, :-1]], axis=1)
            wig = np.abs(rng.normal(0, 1, (m, 241))) * (sA[:, None] / np.sqrt(240)) / 3
            highs = np.round(np.minimum(np.maximum(opens, closes) * (1 + wig), up_c[A, None]), 2)
            lows = np.round(np.maximum(np.minimum(opens, closes) * (1 - wig), dn_c[A, None]), 2)

            day_vol = turn_rate[A] * float_sh[A] * np.exp(rng.normal(0, 0.4, m)) * (1 + 10 * np.abs(r_on))
            vol = day_vol[:, None] * prof[None, :] * np.exp(rng.normal(0, 0.5, (m, 241)))
            vol[:, 201:231] *= np.exp(0.3 * b_t)[:, None]                # 14:21-14:50
            at_limit = (closes >= up_c[A, None]) | (closes <= dn_c[A, None])
            vol[at_limit] *= 0.15
            vol[auction, 238:240] = 0.0
            vol = np.round(vol / 100) * 100
            zero = vol == 0
            opens[zero] = closes[zero]
            highs[zero] = closes[zero]
            lows[zero] = closes[zero]
            vwap = (opens + highs + lows + closes) / 4
            amount = np.round(vol * vwap, 2)

            date_us = np.int64((datetime(d.year, d.month, d.day) - datetime(1970, 1, 1)).total_seconds()) * 1_000_000
            ts = (date_us + bar_us)[None, :].repeat(m, 0)
            minute_buf.append(pl.DataFrame({
                "symbol": np.repeat(syms[A], 241),
                "date": pl.Series([d] * (m * 241), dtype=pl.Date),
                "ts": pl.Series(ts.ravel(), dtype=pl.Int64).cast(pl.Datetime("us")),
                "bidx": np.tile(bidx, m),
                "open": opens.ravel(), "high": highs.ravel(), "low": lows.ravel(),
                "close": closes.ravel(), "volume": vol.ravel(), "amount": amount.ravel(),
            }))

            dvol = vol.sum(1)
            for j, i in enumerate(A):
                daily_buf.append({
                    "symbol": syms[i], "date": d, "open": float(p0[j]),
                    "high": float(highs[j].max()), "low": float(lows[j].min()),
                    "close": float(closes[j, 240]), "pre_close": float(pre_close[i]),
                    "volume": float(dvol[j]), "amount": float(amount[j].sum()),
                    "turnover": float(dvol[j] / float_sh[i] * 100), "adj_factor": float(adj[i]),
                    "up_limit": None if np.isnan(up[i]) else float(up[i]),
                    "down_limit": None if np.isnan(dn[i]) else float(dn[i]),
                    "is_suspended": False, "is_st": bool(is_st[i]),
                    "float_mv": float(closes[j, 240] * float_sh[i]),
                    "total_mv": float(closes[j, 240] * total_sh[i]),
                })
            prev_close[A] = closes[:, 240]
            sig_a[A], sig_b[A], sig_c[A] = a_t, b_t, c_t
            gap_drift[A] = 0.0
            days_listed[A] += 1

        for i in np.where(suspended)[0]:
            daily_buf.append({
                "symbol": syms[i], "date": d, "open": float(pre_close[i]), "high": float(pre_close[i]),
                "low": float(pre_close[i]), "close": float(pre_close[i]), "pre_close": float(pre_close[i]),
                "volume": 0.0, "amount": 0.0, "turnover": 0.0, "adj_factor": float(adj[i]),
                "up_limit": None if np.isnan(up[i]) else float(up[i]),
                "down_limit": None if np.isnan(dn[i]) else float(dn[i]),
                "is_suspended": True, "is_st": bool(is_st[i]),
                "float_mv": float(pre_close[i] * float_sh[i]), "total_mv": float(pre_close[i] * total_sh[i]),
            })
            prev_close[i] = pre_close[i]
            gap_drift[i] += beta[i] * rng.normal(0, 0.012) + rng.normal(0, sigma[i])
            sig_a[i] = sig_b[i] = sig_c[i] = 0.0
        susp_left[suspended] -= 1

    flush_month()
    flush_year()
    info = {"dataset": dataset, "n_stocks": n, "n_days": len(cal),
            "minute_rows": n_minute_rows, "seconds": round(_time.time() - t0, 1)}
    if verbose:
        print(f"  sim done: {info}")
    return info
