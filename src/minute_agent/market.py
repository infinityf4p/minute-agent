"""Load market rule configs (configs/markets/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

import yaml

from minute_agent.data.schema import bar_index_at
from minute_agent.paths import REPO_ROOT


@dataclass(frozen=True)
class MarketRules:
    raw: dict[str, Any]

    @property
    def market(self) -> str:
        return self.raw["market"]

    @property
    def decision_time(self) -> str:
        return self.raw["decision_time"]

    def decision_bidx(self, decision_time: str | None = None) -> int:
        return bar_index_at(self.market, decision_time or self.decision_time)

    @property
    def short_selling(self) -> bool:
        return bool(self.raw["short_selling"])

    @property
    def portfolio(self) -> dict[str, Any]:
        return self.raw["portfolio"]

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw.get("universe", {})

    def buy_cost(self, d: date | None = None) -> float:
        c = self.raw["costs"]
        return c.get("commission", 0.0) + c.get("slippage", 0.0)

    def sell_cost(self, d: date) -> float:
        c = self.raw["costs"]
        cost = c.get("commission", 0.0) + c.get("slippage", 0.0) + c.get("sec_fee_sell", 0.0)
        cost += _schedule_value(c.get("stamp_duty_sell"), d, "rate")
        return cost


def _schedule_value(sched, d: date, key: str) -> float:
    if sched is None:
        return 0.0
    if isinstance(sched, (int, float)):
        return float(sched)
    for item in sched:
        lo = date.fromisoformat(str(item["from"])) if "from" in item else date.min
        hi = date.fromisoformat(str(item["until"])) if "until" in item else date.max
        if lo <= d <= hi:
            return float(item[key])
    return 0.0


@lru_cache(maxsize=None)
def load_market(market: str) -> MarketRules:
    path = REPO_ROOT / "configs" / "markets" / f"{market}.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw.get("market") != market:
        raise ValueError(f"{path} declares market={raw.get('market')!r}")
    return MarketRules(raw)
