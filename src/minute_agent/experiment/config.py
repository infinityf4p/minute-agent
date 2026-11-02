"""Experiment config (PLAN.md section 4, v1 action space): load and validate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from minute_agent.features.cache import normalize_spec
from minute_agent.models import MODELS
from minute_agent.panel import TARGETS

ALLOWED_KEYS = {
    "name", "market", "dataset", "target", "universe_filter", "features",
    "cross_section_norm", "model", "train", "eval_split", "seed", "label_norm", "decision_time",
}
DEFAULTS = {
    "market": "cn",
    "dataset": "sim_cn",
    "cross_section_norm": "rank",
    "label_norm": "rank",
    "eval_split": "valid",
    "seed": 0,
    "universe_filter": {},
    "train": {"window_years": 3, "retrain_every": "quarter"},
}
RETRAIN = {"month", "quarter", "half", "year", "never"}


class ConfigError(ValueError):
    pass


def load(path_or_dict) -> dict:
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict(path_or_dict)
    return validate(cfg)


def validate(cfg: dict) -> dict:
    unknown = set(cfg) - ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    out = {**DEFAULTS, **cfg}
    out["train"] = {**DEFAULTS["train"], **cfg.get("train", {})}
    if out["market"] != "cn":
        raise ConfigError("M1 only supports market=cn")
    if out["target"] not in TARGETS:
        raise ConfigError(f"target must be one of {TARGETS}")
    if out["eval_split"] != "valid":
        raise ConfigError("eval_split must be 'valid'; the locked test set is not reachable from configs")
    if not out.get("features"):
        raise ConfigError("features is empty")
    for s in out["features"]:
        if not isinstance(s, dict) or "op" not in s:
            raise ConfigError(f"bad feature spec {s!r}")
        normalize_spec(s)
    if out.get("model", {}).get("type") not in MODELS:
        raise ConfigError(f"model.type must be one of {sorted(MODELS)}")
    if out["train"]["retrain_every"] not in RETRAIN:
        raise ConfigError(f"train.retrain_every must be one of {sorted(RETRAIN)}")
    wy = out["train"]["window_years"]
    if not (0 < float(wy) <= 6):
        raise ConfigError("train.window_years must be in (0, 6]")
    uf = out["universe_filter"] or {}
    if set(uf) - {"min_price", "min_adv_pct"}:
        raise ConfigError(f"unknown universe_filter keys {sorted(set(uf) - {'min_price', 'min_adv_pct'})}")
    return out


def config_hash(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if k != "name"}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]
