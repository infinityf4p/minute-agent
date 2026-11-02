"""The v1 action space, shared by the LLM agent and the random-search baseline.

Every config the agent proposes must pass `check(cfg)`; random search samples
from exactly the same domains with `sample(rng)`. `describe()` renders the
space for the LLM prompt.
"""

from __future__ import annotations

import copy
import json
import random

TARGETS = ["overnight", "next_cc"]
NORMS = ["rank", "zscore", "none"]
LABEL_NORMS = ["rank", "raw"]
RETRAIN = ["month", "quarter", "half", "year"]
WINDOW_YEARS = [1, 2, 3, 4, 5]
MIN_PRICE = [0, 1, 2, 3, 5]
MIN_ADV_PCT = [0, 10, 20, 30, 50]
MAX_FEATURE_SPECS = 10

# op -> {param: ("subset" | "choice", domain)}
OPS: dict[str, dict] = {
    "ret": {"window_min": ("subset", [1, 5, 10, 15, 30, 60, 120, 240])},
    "realized_vol": {"window_min": ("subset", [5, 15, 30, 60, 120, 240])},
    "vwap_dev": {"at": ("choice", ["decision_time"])},
    "volume_profile": {"buckets": ("subset", ["open15", "open30", "open60", "mid", "last5", "last15", "last30", "last60"])},
    "last_n_min_stats": {"n": ("choice", [5, 10, 15, 30, 60])},
    "jump_count": {"threshold_sigma": ("choice", [3, 4, 5])},
    "daily_lag": {"fields": ("subset", ["ret", "turnover"]), "lags": ("subset", [1, 2, 3, 5, 10, 20, 60])},
    "return_moments": {"window_min": ("subset", [30, 60, 120, 240])},
    "pv_corr": {"window_min": ("subset", [30, 60, 120, 240])},
    "price_position": {},
    "signed_volume": {"window_min": ("subset", [5, 15, 30, 60, 120, 240])},
    "daily_stats": {"windows": ("subset", [5, 10, 20, 60, 120])},
}

MODELS: dict[str, dict] = {
    "lgbm": {"num_leaves": [15, 31, 63, 127], "lr": [0.02, 0.05, 0.1], "n_estimators": [100, 200, 500, 1000],
             "min_data_in_leaf": [50, 200, 1000], "feature_fraction": [0.6, 0.8, 1.0]},
    "xgb": {"max_depth": [3, 4, 6, 8], "lr": [0.02, 0.05, 0.1], "n_estimators": [100, 200, 500, 1000],
            "min_child_weight": [10, 50, 200], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]},
    "mlp": {"hidden": [[64], [128, 64], [256, 128], [256, 128, 64]], "dropout": [0.0, 0.1, 0.3],
            "lr": [0.0003, 0.001, 0.003], "epochs": [3, 5, 8, 12], "batch_size": [2048, 4096, 8192]},
    "linear": {"alpha": [0.1, 1.0, 10.0, 100.0]},
}
# M3 adds sequence models; enabled per session
SEQ_MODELS: dict[str, dict] = {
    "gru_seq": {"hidden": [32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.3], "lr": [0.0003, 0.001, 0.003],
                "epochs": [2, 4, 6], "batch_size": [1024, 2048, 4096], "with_features": [True, False]},
    "transformer_seq": {"d_model": [32, 64], "heads": [2, 4], "layers": [1, 2], "dropout": [0.0, 0.1],
                        "lr": [0.0003, 0.001], "epochs": [2, 4, 6], "batch_size": [1024, 2048], "with_features": [True, False]},
}


class SpaceError(ValueError):
    pass


def models(seq: bool = False) -> dict:
    return {**MODELS, **(SEQ_MODELS if seq else {})}


def _check_value(where: str, v, kind: str, dom: list):
    if kind == "choice":
        if v not in dom:
            raise SpaceError(f"{where}={v!r} not in {dom}")
    else:
        if not isinstance(v, list) or not v:
            raise SpaceError(f"{where} must be a non-empty list drawn from {dom}")
        bad = [x for x in v if x not in dom]
        if bad or len(set(map(json.dumps, v))) != len(v):
            raise SpaceError(f"{where}={v!r}: values must be distinct and in {dom}")


def check(cfg: dict, seq_models: bool = False, custom_features: bool = False) -> dict:
    """Validate a config against the action space; return a normalized copy."""
    allowed = {"target", "universe_filter", "features", "cross_section_norm", "label_norm", "model", "train"}
    extra = set(cfg) - allowed - {"name", "seed"}
    if extra:
        raise SpaceError(f"keys outside the action space: {sorted(extra)}")
    c = copy.deepcopy(cfg)
    if c.get("target") not in TARGETS:
        raise SpaceError(f"target must be one of {TARGETS}")
    uf = c.get("universe_filter", {}) or {}
    _check_value("universe_filter.min_price", uf.get("min_price", 0), "choice", MIN_PRICE)
    _check_value("universe_filter.min_adv_pct", uf.get("min_adv_pct", 0), "choice", MIN_ADV_PCT)
    c["universe_filter"] = {"min_price": uf.get("min_price", 0), "min_adv_pct": uf.get("min_adv_pct", 0)}
    feats = c.get("features")
    if not isinstance(feats, list) or not (1 <= len(feats) <= MAX_FEATURE_SPECS):
        raise SpaceError(f"features must be a list of 1..{MAX_FEATURE_SPECS} specs")
    seen = set()
    for f in feats:
        op = f.get("op")
        if op == "custom":
            if not custom_features:
                raise SpaceError("custom features are not enabled in this session")
            if set(f) - {"op", "name", "code"} or not f.get("name") or not f.get("code"):
                raise SpaceError("custom feature needs exactly: op, name, code")
        elif op not in OPS:
            raise SpaceError(f"unknown op {op!r}; allowed: {sorted(OPS)}")
        else:
            params = {k: v for k, v in f.items() if k != "op"}
            if set(params) - set(OPS[op]):
                raise SpaceError(f"{op}: unknown params {sorted(set(params) - set(OPS[op]))}")
            for p, (kind, dom) in OPS[op].items():
                if p not in params:
                    raise SpaceError(f"{op}: missing param {p}")
                _check_value(f"{op}.{p}", params[p], kind, dom)
        key = json.dumps(f, sort_keys=True)
        if key in seen:
            raise SpaceError(f"duplicate feature spec {f}")
        seen.add(key)
    for k, dom in (("cross_section_norm", NORMS), ("label_norm", LABEL_NORMS)):
        _check_value(k, c.get(k), "choice", dom)
    m = c.get("model") or {}
    mdom = models(seq_models)
    if m.get("type") not in mdom:
        raise SpaceError(f"model.type must be one of {sorted(mdom)}")
    params = m.get("params", {}) or {}
    if set(params) - set(mdom[m["type"]]):
        raise SpaceError(f"model.params: unknown {sorted(set(params) - set(mdom[m['type']]))}")
    for p, dom in mdom[m["type"]].items():
        if p in params:
            _check_value(f"model.params.{p}", params[p], "choice", dom)
    tr = c.get("train") or {}
    _check_value("train.window_years", tr.get("window_years"), "choice", WINDOW_YEARS)
    _check_value("train.retrain_every", tr.get("retrain_every"), "choice", RETRAIN)
    return c


def _subset(rng: random.Random, dom: list) -> list:
    k = rng.randint(1, min(len(dom), 4))
    picked = rng.sample(dom, k)
    return [x for x in dom if x in picked]


def sample(rng: random.Random, seq_models: bool = False) -> dict:
    """Uniform-ish random config from the same space."""
    ops = rng.sample(sorted(OPS), rng.randint(2, 7))
    feats = []
    for op in ops:
        spec = {"op": op}
        for p, (kind, dom) in OPS[op].items():
            spec[p] = rng.choice(dom) if kind == "choice" else _subset(rng, dom)
        feats.append(spec)
    mdom = models(seq_models)
    mt = rng.choice(sorted(mdom))
    params = {p: rng.choice(dom) for p, dom in mdom[mt].items()}
    cfg = {
        "target": rng.choice(TARGETS),
        "universe_filter": {"min_price": rng.choice(MIN_PRICE), "min_adv_pct": rng.choice(MIN_ADV_PCT)},
        "features": feats,
        "cross_section_norm": rng.choice(NORMS),
        "label_norm": rng.choice(LABEL_NORMS),
        "model": {"type": mt, "params": params},
        "train": {"window_years": rng.choice(WINDOW_YEARS), "retrain_every": rng.choice(RETRAIN)},
    }
    return check(cfg, seq_models=seq_models)


def describe(seq_models: bool = False, custom_features: bool = False) -> str:
    lines = ["Config keys and allowed values (lists = choose a non-empty subset in the given order-free way):",
             f"- target: one of {TARGETS}",
             f"- universe_filter: {{min_price: one of {MIN_PRICE}, min_adv_pct: one of {MIN_ADV_PCT}}}",
             f"- features: list of 1..{MAX_FEATURE_SPECS} op specs, each {{op: <name>, <params>}}:"]
    for op, ps in OPS.items():
        desc = ", ".join(f"{p}: {'subset of' if k == 'subset' else 'one of'} {dom}" for p, (k, dom) in ps.items())
        lines.append(f"    - {op}" + (f" ({desc})" if desc else " (no params)"))
    if custom_features:
        lines.append("    - custom (name: str, code: python source defining `feature(m, daily, d)`; see CUSTOM FEATURES)")
    lines += [f"- cross_section_norm: one of {NORMS}",
              f"- label_norm: one of {LABEL_NORMS}",
              "- model: {type: <name>, params: {...}}; every param optional, defaults used if omitted:"]
    for mt, ps in models(seq_models).items():
        lines.append(f"    - {mt}: " + ", ".join(f"{p} in {dom}" for p, dom in ps.items()))
    lines.append(f"- train: {{window_years: one of {WINDOW_YEARS}, retrain_every: one of {RETRAIN}}}")
    return "\n".join(lines)
