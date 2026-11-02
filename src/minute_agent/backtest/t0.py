"""Base-position T0 overlay (template T2, and the overlay part of T4).

Given a base holding per stock-day (value at the previous close, all sellable),
the overlay trades around it and returns the base to its original size:
  sell_first (反T): sell part of the base, buy it back later
  buy_first  (正T): buy extra, sell the same amount of old base shares later
Modes:
  intraday   entries and exits at the decision points; everything is flattened
             from `flatten_at` (default 14:45), retried at every later decision point
             (e.g. 14:50, 14:55) with partial fills under the participation cap; what
             still can't be flattened is settled at the next opening auction
  overnight  sell part of the base after the decision near the close, buy it
             back at the next opening auction

PnL is measured against simply holding the base: for every round trip
shares x (sell price - buy price) - costs. Results are in currency and in bp of
the total base value.

Entries use same-time cross-sectional quantiles of the score across the base
stocks (causal). The engine enforces: tradability flags, participation cap,
quantity <= sellable base, forced flatten, costs.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from minute_agent.backtest.costs import CostModel

KEYS = ["symbol", "date"]


def simulate_t0(base: pl.DataFrame, signals: pl.DataFrame, exe: pl.DataFrame, auction: pl.DataFrame,
                params: dict, costs: CostModel, position_sizing: bool = False) -> tuple[pl.DataFrame, dict]:
    """base: symbol, date, base_value
    signals: symbol, date, d, score     (d = decision bar index)
    exe: symbol, date, d, px, avail, can_buy, can_sell, sigma
    auction: symbol, date, open_px, open_can_buy, open_can_sell, close_px (next-day lookups)
    params: mode, direction, size_frac, max_trips, entry_q, hold_bars, take_profit, stop_loss,
            signal_exit, flatten_at"""
    mode = params.get("mode", "intraday")
    direction = params.get("direction", "both")
    f = float(params.get("size_frac", 0.25))
    max_trips = int(params.get("max_trips", 1))
    q = float(params.get("entry_q", 0.9))
    hold = params.get("hold_bars")
    tp, sl = params.get("take_profit"), params.get("stop_loss")
    sig_exit = bool(params.get("signal_exit", False))
    flatten_at = int(params.get("flatten_at", 225))
    cap_part = costs.participation_cap

    s = signals.join(base, on=KEYS, how="inner").join(exe, on=KEYS + ["d"], how="left")
    s = s.with_columns(
        pl.col("score").quantile(1 - q).over("date", "d").alias("q_lo"),
        pl.col("score").quantile(q).over("date", "d").alias("q_hi"),
        pl.col("score").median().over("date", "d").alias("q_mid"),
        pl.col("score").std().over("date", "d").alias("q_sd"),
    )
    # next trading day's opening auction for each stock-day (settlement of carries / overnight buy-back)
    nxt = auction.sort("symbol", "date").with_columns(
        pl.col("open_px").shift(-1).over("symbol").alias("n_open"),
        pl.col("open_can_buy").shift(-1).over("symbol").alias("n_can_buy"),
        pl.col("open_can_sell").shift(-1).over("symbol").alias("n_can_sell"),
        pl.col("close_px").shift(-1).over("symbol").alias("n_close"),
    ).select(KEYS + ["n_open", "n_can_buy", "n_can_sell", "n_close"])
    sd = base.join(nxt, on=KEYS, how="left").sort(KEYS)
    idx = {k: i for i, k in enumerate(zip(sd["symbol"].to_list(), sd["date"].to_list()))}
    M = sd.height
    base_v = sd["base_value"].to_numpy().astype(float)
    dates = np.array(sd["date"].to_list(), dtype="datetime64[D]")

    state = np.zeros(M, dtype=int)        # 0 flat, -1 sold first (owe a buy), +1 bought first (owe a sell)
    shares = np.zeros(M)
    entry_px = np.full(M, np.nan)
    entry_d = np.full(M, -1)
    trips = np.zeros(M, dtype=int)
    pnl = np.zeros(M)
    cost = np.zeros(M)
    n_trades = np.zeros(M, dtype=int)
    wins = np.zeros(M, dtype=int)
    closed = np.zeros(M, dtype=int)

    leg_pnl = np.zeros(M)   # pnl accumulated by partial exits of the current round trip

    def close_out(rows, price, ok, max_value=None):
        """Exit (part of) the open leg. With max_value, only that much value is traded now;
        the rest stays open and is retried at the next decision point."""
        rows, price = rows[ok], price[ok]
        if len(rows) == 0:
            return
        qty = shares[rows].copy()
        if max_value is not None:
            qty = np.minimum(qty, max_value[ok] / price)
        val = qty * price
        is_sell = state[rows] == 1
        c = costs.trade_cost(val, is_sell, dates[rows])
        gain = np.where(state[rows] == -1, qty * (entry_px[rows] - price), qty * (price - entry_px[rows]))
        pnl[rows] += gain
        cost[rows] += c
        leg_pnl[rows] += gain - c
        n_trades[rows] += 1
        shares[rows] -= qty
        done = shares[rows] <= 1e-9
        r = rows[done]
        wins[r] += (leg_pnl[r] > 0)
        closed[r] += 1
        state[r] = 0
        shares[r] = 0.0
        trips[r] += 1
        leg_pnl[r] = 0.0

    for d in sorted(s["d"].unique().to_list()):
        x = s.filter(pl.col("d") == d)
        rows = np.array([idx[k] for k in zip(x["symbol"].to_list(), x["date"].to_list())])
        px = x["px"].to_numpy().astype(float)
        has = np.isfinite(px)
        can_buy = x["can_buy"].fill_null(False).to_numpy() & has
        can_sell = x["can_sell"].fill_null(False).to_numpy() & has
        avail_val = np.nan_to_num(x["avail"].to_numpy().astype(float)) * np.nan_to_num(px) * cap_part
        sc = x["score"].to_numpy().astype(float)
        lo, hi, mid, sdv = (x[c].to_numpy().astype(float) for c in ("q_lo", "q_hi", "q_mid", "q_sd"))

        # ---- exits
        st = state[rows]
        if mode == "intraday":
            open_ = st != 0
            ret = np.where(st == -1, entry_px[rows] / np.where(has, px, np.nan) - 1, np.where(has, px, np.nan) / entry_px[rows] - 1)
            want_exit = open_ & (d >= flatten_at)
            if hold is not None:
                want_exit |= open_ & (d - entry_d[rows] >= int(hold))
            if tp is not None:
                want_exit |= open_ & (np.nan_to_num(ret, nan=0.0) >= float(tp))
            if sl is not None:
                want_exit |= open_ & (np.nan_to_num(ret, nan=0.0) <= -float(sl))
            if sig_exit:
                want_exit |= open_ & (((st == -1) & (sc >= mid)) | ((st == 1) & (sc <= mid)))
            ok = want_exit & np.where(st == -1, can_buy, can_sell) & (avail_val > 0)
            close_out(rows, px, ok, max_value=avail_val)      # partial fills allowed, rest retried later

        # ---- entries
        st = state[rows]
        allow = (st == 0) & (trips[rows] < max_trips) & np.isfinite(sc)
        if mode == "intraday":
            allow &= d < flatten_at
        size = f * base_v[rows]
        if position_sizing:
            z = np.abs(sc - mid) / np.where(sdv > 0, sdv, np.nan)
            size = size * np.clip(np.nan_to_num(z) / 2.0, 0.25, 1.0)
        size = np.minimum(size, avail_val)
        sell_first = allow & (direction in ("sell_first", "both")) & (sc <= lo) & can_sell & (size > 0)
        buy_first = allow & (direction in ("buy_first", "both")) & (sc >= hi) & can_buy & (size > 0) & ~sell_first
        if mode == "overnight":
            buy_first[:] = False
        for mask, sign in ((sell_first, -1), (buy_first, 1)):
            r = rows[mask]
            if len(r) == 0:
                continue
            p = px[mask]
            shares[r] = size[mask] / p
            entry_px[r] = p
            entry_d[r] = d
            state[r] = sign
            c = costs.trade_cost(size[mask], np.full(len(r), sign == -1), dates[r])
            cost[r] += c
            leg_pnl[r] = -c          # a round trip counts as a win only if it beats both legs' costs
            n_trades[r] += 1

    # ---- settle what is still open at the next opening auction (overnight exposure)
    carried = state != 0
    n_open = sd["n_open"].to_numpy().astype(float)
    n_close = sd["n_close"].to_numpy().astype(float)
    ok_open = np.where(state == -1, sd["n_can_buy"].fill_null(False).to_numpy(),
                       sd["n_can_sell"].fill_null(False).to_numpy()) & np.isfinite(n_open)
    settle = np.where(ok_open, n_open, n_close)          # next-day close if the open is locked
    has_settle = np.isfinite(settle)
    carry_rows = np.where(carried & has_settle)[0]
    if mode == "intraday":
        n_failed = int(carried.sum())
    else:
        n_failed = int((carried & ~ok_open).sum())
    close_out(carry_rows, settle[carry_rows], np.ones(len(carry_rows), bool))
    # rows with no next-day price at all (end of sample): drop their open legs at entry (no pnl)

    sd = sd.with_columns(pl.Series("pnl", pnl), pl.Series("cost", cost), pl.Series("trades", n_trades),
                         pl.Series("wins", wins), pl.Series("round_trips", closed))
    daily = sd.group_by("date").agg(
        (pl.col("pnl").sum() - pl.col("cost").sum()).alias("net"),
        pl.col("pnl").sum().alias("gross"), pl.col("cost").sum().alias("cost_cny"),
        pl.col("base_value").sum().alias("base"), pl.col("round_trips").sum().alias("round_trips"),
        pl.col("wins").sum().alias("wins"),
    ).sort("date").with_columns(
        (pl.col("net") / pl.col("base")).alias("overlay_net"),
        (pl.col("gross") / pl.col("base")).alias("overlay_gross"),
        (pl.col("cost_cny") / pl.col("base")).alias("overlay_cost"),
    )
    rt = int(closed.sum())
    summary = {
        "round_trips": rt,
        "round_trips_per_stock_day": rt / max(M, 1),
        "win_rate": float(wins.sum() / max(rt, 1)),
        "carried_overnight_frac": float(carried.sum() / max(rt, 1)) if mode == "intraday" else None,
        "failed_settlement": n_failed,
    }
    return daily, summary
