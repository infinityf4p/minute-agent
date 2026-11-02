"""Statistics for the final report: Sharpe tests, deflated Sharpe ratio, Holm correction."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

N = NormalDist()
EULER = 0.5772156649015329


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else float("nan")


def t_test_mean(x: np.ndarray) -> tuple[float, float]:
    """One-sided test of mean > 0 (normal approximation). Returns (t, p)."""
    x = x[np.isfinite(x)]
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return float(t), float(1 - N.cdf(t))


def moments(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z ** 3).mean()), float((z ** 4).mean())


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies (Bailey & Lopez de Prado)."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER) * N.inv_cdf(1 - 1 / n_trials)
                                + EULER * N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(x: np.ndarray, n_trials: int, var_sr_trials: float) -> dict:
    """Probability that the true Sharpe of returns x exceeds the best Sharpe expected
    from n_trials skill-less trials. Sharpe values are per period (daily)."""
    x = x[np.isfinite(x)]
    T = len(x)
    sr = sharpe(x)
    skew, kurt = moments(x)
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    dsr = N.cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return {"sr_daily": sr, "sr_annual": sr * math.sqrt(252), "sr0_daily": sr0, "sr0_annual": sr0 * math.sqrt(252),
            "n_trials": n_trials, "T": T, "skew": skew, "kurt": kurt, "dsr": dsr}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    return out
