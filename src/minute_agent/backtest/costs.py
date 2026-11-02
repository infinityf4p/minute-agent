"""Trading cost model, configured by hand in configs/costs/*.yaml (never by the agent)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import yaml

from minute_agent.paths import REPO_ROOT


def _schedule(sched, d: date) -> float:
    if sched is None:
        return 0.0
    if isinstance(sched, (int, float)):
        return float(sched)
    for item in sched:
        lo = date.fromisoformat(str(item["from"])) if "from" in item else date.min
        hi = date.fromisoformat(str(item["until"])) if "until" in item else date.max
        if lo <= d <= hi:
            return float(item["rate"])
    return 0.0


@dataclass(frozen=True)
class CostModel:
    raw: dict

    @classmethod
    def load(cls, name: str = "default") -> "CostModel":
        with open(REPO_ROOT / "configs" / "costs" / f"{name}.yaml", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    @property
    def name(self) -> str:
        return self.raw.get("name", "custom")

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()[:10]

    @property
    def participation_cap(self) -> float:
        return float(self.raw.get("participation_cap", 0.1))

    def stamp(self, d: date) -> float:
        return _schedule(self.raw.get("stamp_duty_sell"), d)

    def stamp_array(self, dates: np.ndarray) -> np.ndarray:
        """Vectorized sell-side stamp duty for an array of numpy datetime64[D] dates."""
        out = np.zeros(len(dates))
        sched = self.raw.get("stamp_duty_sell")
        if isinstance(sched, (int, float)):
            return out + float(sched)
        for item in sched or []:
            lo = np.datetime64(str(item.get("from", "1900-01-01")))
            hi = np.datetime64(str(item.get("until", "2999-12-31")))
            out[(dates >= lo) & (dates <= hi)] = float(item["rate"])
        return out

    def trade_cost(self, notional: np.ndarray, is_sell: np.ndarray, dates: np.ndarray,
                   sigma_bar: np.ndarray | None = None, participation: np.ndarray | None = None) -> np.ndarray:
        """Cost in currency for each trade (vectorized). notional >= 0."""
        r = self.raw
        notional = np.abs(np.asarray(notional, dtype=float))
        traded = notional > 0
        commission = np.maximum(notional * r.get("commission_rate", 0.0), r.get("min_commission", 0.0)) * traded
        fees = notional * r.get("transfer_fee_rate", 0.0)
        stamp = notional * self.stamp_array(dates) * np.asarray(is_sell, dtype=bool)
        slip = notional * r.get("slippage_bps", 0.0) / 1e4
        impact = 0.0
        y = r.get("impact_coef", 0.0)
        if y and sigma_bar is not None and participation is not None:
            impact = notional * y * np.nan_to_num(sigma_bar) * np.sqrt(np.clip(np.nan_to_num(participation), 0, None))
        return (commission + fees + stamp + slip + impact) * r.get("cost_multiplier", 1.0)

    def round_trip_bps(self, d: date, notional: float = 1e6) -> float:
        """Cost of buying and selling `notional` once, in bp of notional (for reports)."""
        c = self.trade_cost(np.array([notional, notional]), np.array([False, True]),
                            np.array([np.datetime64(d)] * 2))
        return float(c.sum() / notional * 1e4)
