"""Study configurations. Thresholds are fixed before any search result is seen; their hash is stored in
every session so a later change is visible."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from minute_agent.paths import cache_root, runs_root

# Admission rules (evaluator only; the LLM never scores factors)
THRESHOLDS = {
    "min_coverage": 0.8,        # share of eligible stock-days with a factor value
    # calibrated once on an 8-candidate random smoke run (t >= 3, corr < 0.7 admitted 6 of 8), before any arm
    # comparison: tightened to the A-share redundancy bound of FactorMiner (0.5) and t >= 4
    "min_res_t": 4.0,           # |t| of the mean daily residual rank IC over the mining period
    "fold_sign": True,          # every mining fold has the same sign of mean residual rank IC
    "max_corr": 0.5,            # max |time-averaged cross-sectional rank corr| with library members
    "replace_ratio": 1.3,       # replace the single conflicting member if |res IC| is this much larger
    "min_stocks_per_date": 50,
    "seed_reference": True,     # the library starts with the classic reference factors (fixed, not replaceable)
}
COST_BP_ONE_WAY = 12.0          # rough one-way cost for the long-short decile spread (commission+stamp+slippage)

CONFIGS = {
    "pilot": {
        "dataset": "cn",
        "n_symbols": 600,               # stratified by ADV terciles at the reference window
        "universe_seed": 0,
        "universe_ref": (date(2019, 10, 1), date(2019, 12, 31)),
        "mine": [(date(2020, 1, 1), date(2020, 12, 31)), (date(2021, 1, 1), date(2021, 12, 31))],
        "valid": [(date(2022, 1, 1), date(2022, 12, 31)), (date(2023, 1, 1), date(2023, 12, 31))],
        "warmup_months": 4,             # intraday values before a period, for time-series operators
        "decision_bar": 230,            # 14:50
        "min_listed_days": 120,
    },
    "full": {
        "dataset": "cn",
        "n_symbols": None,              # all eligible stocks
        "universe_seed": 0,
        "universe_ref": (date(2017, 10, 1), date(2017, 12, 31)),
        "mine": [(date(y, 1, 1), date(y, 12, 31)) for y in (2018, 2019, 2020, 2021)],
        "valid": [(date(2022, 1, 1), date(2022, 12, 31)), (date(2023, 1, 1), date(2023, 12, 31))],
        "warmup_months": 4,
        "decision_bar": 230,
        "min_listed_days": 120,
    },
}


def get(name: str) -> dict:
    return {**CONFIGS[name], "name": name}


def fingerprint(cfg: dict) -> str:
    blob = json.dumps({"cfg": cfg, "thresholds": THRESHOLDS, "cost": COST_BP_ONE_WAY}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def cache_dir(cfg: dict) -> Path:
    p = cache_root() / "factormine" / cfg["name"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    p = runs_root() / "factormine"
    p.mkdir(parents=True, exist_ok=True)
    return p


def span(periods: list[tuple[date, date]]) -> tuple[date, date]:
    return min(a for a, _ in periods), max(b for _, b in periods)
