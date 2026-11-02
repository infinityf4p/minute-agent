"""Multi-day long-only book for templates T1 (cross-section, multi-day hold) and
T3 (enter after the decision, exit at the next opening auction).

Each day has four price points: previous close -> open (bar 0) -> execution
price after the decision bar -> close. Positions are carried as values and
re-priced segment by segment (adjusted for ex-dates on the first segment).
Hedged excess return = sum over segments of position value x (stock return -
equal-weight universe return over the same segment), minus costs. This is the
"hedge with an index future" proxy (no basis) and works for overnight holds.

Rules enforced here (the agent cannot change them):
* trades happen only at the execution price of the decision bar (or the
  opening auction for T3 exits), never at the signal bar
* buys need can_buy, sells need can_sell (limit-locked bars don't fill)
* participation cap on the execution bars' volume
* T+1: positions entered today are not sold today
* suspended or untraded names keep their position
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from minute_agent.backtest.costs import CostModel


@dataclass
class Dense:
    dates: np.ndarray        # datetime64[D], shape [T]
    symbols: list[str]       # shape [N]
    arrays: dict[str, np.ndarray]


def densify(frames: dict[str, tuple[pl.DataFrame, list[str]]], dates: list, symbols: list[str]) -> Dense:
    """Pivot long frames (date, symbol, cols...) into [T, N] arrays."""
    di = {d: i for i, d in enumerate(dates)}
    si = {s: j for j, s in enumerate(symbols)}
    T, N = len(dates), len(symbols)
    out = {}
    for _, (df, cols) in frames.items():
        if df.height == 0:
            for c in cols:
                out[c] = np.full((T, N), np.nan)
            continue
        r = np.array([di.get(d, -1) for d in df["date"].to_list()])
        c_ = np.array([si.get(s, -1) for s in df["symbol"].to_list()])
        ok = (r >= 0) & (c_ >= 0)
        for c in cols:
            a = np.full((T, N), np.nan)
            v = df[c].cast(pl.Float64).to_numpy()
            a[r[ok], c_[ok]] = v[ok]
            out[c] = a
    return Dense(np.array(dates, dtype="datetime64[D]"), symbols, out)


def _weights(score_row, held, n_sel, buffer, method, sigma_row, max_w, buyable_now=None):
    """Target selection with a no-trade buffer, then weights. Returns weight vector (sums to 1 or 0).
    New names must be buyable at the decision time (not at the up limit then); whether the
    order actually fills is decided later by the execution bars."""
    N = len(score_row)
    elig = np.isfinite(score_row)
    if not elig.any() or n_sel <= 0:
        return np.zeros(N)
    order = np.argsort(-np.where(elig, score_row, -np.inf))
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    keep = held & elig & (rank < n_sel + buffer)          # held names inside the buffer stay
    slots = max(0, n_sel - keep.sum())
    new = (~keep) & elig & ~held
    if buyable_now is not None:
        new &= buyable_now
    cand = [j for j in order if new[j]][:slots]
    sel = keep.copy()
    sel[cand] = True
    if not sel.any():
        return np.zeros(N)
    if method == "equal":
        w = sel.astype(float)
    elif method == "score":
        s = np.where(sel, n_sel + buffer - rank, 0.0).astype(float)
        w = np.clip(s, 0, None)
    elif method == "inv_vol":
        sg = np.where(sel, sigma_row, np.nan)
        med = np.nanmedian(sg) if np.isfinite(sg).any() else 1.0
        w = np.where(sel, 1.0 / np.clip(np.nan_to_num(sg, nan=med), 1e-5, None), 0.0)
    else:
        raise ValueError(f"unknown weighting {method!r}")
    w = w / w.sum()
    for _ in range(10):                                      # cap and redistribute
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        under = (w > 0) & ~over
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / max(w.sum(), 1e-12)


def simulate_book(dense: Dense, params: dict, costs: CostModel, capital: float = 1e7,
                  position_sizing: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Arrays needed in dense.arrays (shape [T, N]):
        score, in_univ, buyable_now (known at the decision), close, adj, open_px, px, avail,
        can_buy, can_sell (execution bars), open_can_sell, sigma
    params: mode ('hold' | 'next_open'), top_frac or top_k, buffer_frac, weighting, max_weight,
            rebalance_every, hold_min, turnover_cap, stop_loss, exposure_rule, target_vol
    Returns (daily results, positions at each day's open for T4)."""
    A = dense.arrays
    T, N = A["close"].shape
    mode = params.get("mode", "hold")
    weighting = params.get("weighting", "equal") if position_sizing else "equal"
    max_w = float(params.get("max_weight", 0.05))
    reb = int(params.get("rebalance_every", 1))
    hold_min = int(params.get("hold_min", 1))
    tcap = params.get("turnover_cap")
    stop = params.get("stop_loss")
    cap_part = costs.participation_cap

    V = np.zeros(N)          # position values
    age = np.zeros(N, dtype=int)
    entry = np.full(N, np.nan)  # entry price (adjusted) for stop loss
    cash = capital
    rows, pos_rows = [], []
    eq_hist = []
    prev_close = A["close"][0].copy()
    prev_adj = A["adj"][0].copy()

    for t in range(T):
        d = dense.dates[t]
        close, adj = A["close"][t], A["adj"][t]
        op = np.where(np.isfinite(A["open_px"][t]), A["open_px"][t], close)
        px = np.where(np.isfinite(A["px"][t]), A["px"][t], close)
        has_px = np.isfinite(A["px"][t])
        univ = A["in_univ"][t] > 0
        E0 = cash + V.sum()
        if t > 0:
            pos_rows.append((d, V.copy(), age.copy()))
        hedged = 0.0
        # segment 1: prev close -> open (ex-date adjustment here)
        r1 = np.where(np.isfinite(prev_close) & np.isfinite(op), op * adj / (prev_close * prev_adj) - 1, 0.0)
        r1 = np.nan_to_num(r1)
        U1 = float(np.nanmean(r1[univ])) if univ.any() else 0.0
        hedged += float((V * (r1 - U1)).sum())
        V = V * (1 + r1)
        cost_today = 0.0
        traded = 0.0
        # T3: everything bought yesterday is sold at the opening auction
        if mode == "next_open":
            can = (A["open_can_sell"][t] > 0) & (V > 0)
            vol_cap = np.nan_to_num(A["avail_open"][t]) * op * cap_part if "avail_open" in A else np.full(N, np.inf)
            sell = np.where(can, np.minimum(V, vol_cap), 0.0)
            c = costs.trade_cost(sell, np.ones(N, bool), np.full(N, d))
            cash += sell.sum() - c.sum()
            V = V - sell
            cost_today += c.sum()
            traded += sell.sum()
        # segment 2: open -> execution price
        r2 = np.nan_to_num(px / op - 1)
        U2 = float(np.nanmean(r2[univ])) if univ.any() else 0.0
        hedged += float((V * (r2 - U2)).sum())
        V = V * (1 + r2)
        # decision: stop loss, rebalance
        E_dec = cash + V.sum()
        score = np.where(univ, A["score"][t], np.nan)
        orders = np.zeros(N)
        if stop is not None:
            hit = (V > 0) & (age >= 1) & np.isfinite(entry) & (px * adj / entry - 1 < -float(stop))
            orders[hit] = -V[hit]
        if mode == "next_open" or t % reb == 0:
            n_univ = int(np.isfinite(score).sum())
            n_sel = int(params["top_k"]) if "top_k" in params else max(1, int(round(float(params.get("top_frac", 0.1)) * n_univ)))
            buffer = int(round(float(params.get("buffer_frac", 0.0)) * n_sel))
            held = V > 0
            bnow = (A["buyable_now"][t] > 0) if "buyable_now" in A else None
            w = _weights(score, held, n_sel, buffer, weighting, A["sigma"][t], max_w, bnow)
            expo = 1.0
            if position_sizing:
                rule = params.get("exposure_rule", "full")
                if rule == "vol_target" and len(eq_hist) > 20:
                    rv = np.std(np.diff(np.log(eq_hist[-21:]))) * np.sqrt(252)
                    expo = float(np.clip(params.get("target_vol", 0.15) / max(rv, 1e-6), 0.2, 1.0))
                elif rule == "trend" and t > 20:
                    expo = 1.0 if np.nanmean(A["univ_ret20"][t]) >= 0 else 0.5
            target = w * expo * E_dec
            want = target - V
            locked = (age < hold_min) & (V > 0) & (want < 0)       # minimum holding period
            want[locked] = 0.0
            orders = np.where(orders != 0, orders, want)
        # tradability and T+1 (age 0 = bought today, cannot be sold today)
        buy_ok = (A["can_buy"][t] > 0) & has_px
        sell_ok = (A["can_sell"][t] > 0) & has_px & (age >= 1)
        orders = np.where(orders > 0, np.where(buy_ok, orders, 0.0), np.where(sell_ok, orders, 0.0))
        orders = np.where(orders < 0, np.maximum(orders, -V), orders)
        cap_val = np.nan_to_num(A["avail"][t]) * px * cap_part
        orders = np.clip(orders, -cap_val, cap_val)
        if tcap is not None and np.abs(orders).sum() > float(tcap) * E_dec:
            orders *= float(tcap) * E_dec / np.abs(orders).sum()
        buys = np.clip(orders, 0, None)
        # don't buy more than the cash left after sells
        avail_cash = cash + np.clip(-orders, 0, None).sum()
        if buys.sum() > avail_cash:
            orders = np.where(orders > 0, orders * avail_cash / buys.sum(), orders)
        part = np.abs(orders) / np.clip(np.nan_to_num(A["avail"][t]) * px, 1e-9, None)
        c = costs.trade_cost(orders, orders < 0, np.full(N, d), A["sigma"][t], part)
        cash -= orders.sum() + c.sum()
        cost_today += c.sum()
        traded += np.abs(orders).sum()
        new_pos = (V <= 1e-9) & (orders > 0)
        V = V + orders
        age = np.where(V > 1e-9, np.where(new_pos, 0, age), 0)
        entry = np.where(new_pos, px * adj, np.where(V > 1e-9, entry, np.nan))
        # segment 3: execution price -> close
        r3 = np.nan_to_num(close / px - 1)
        U3 = float(np.nanmean(r3[univ])) if univ.any() else 0.0
        hedged += float((V * (r3 - U3)).sum())
        V = V * (1 + r3)
        E1 = cash + V.sum()
        rows.append({
            "date": d.astype(object), "equity": E1, "ret_net": E1 / E0 - 1,
            "excess_net": (hedged - cost_today) / E0, "excess_gross": hedged / E0,
            "cost": cost_today / E0, "turnover": traded / E0, "exposure": V.sum() / E1,
            "n_pos": int((V > 1e-9).sum()),
        })
        eq_hist.append(E1)
        age = np.where(V > 1e-9, age + 1, 0)
        prev_close, prev_adj = close, adj
    daily = pl.DataFrame(rows)
    return daily, pos_rows
