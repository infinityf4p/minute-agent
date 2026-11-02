"""v2 action space: the whole trading workflow (signal, target, model, strategy
template, portfolio/T0 parameters, execution style), gated by human session switches.

Session settings (set by a person, never by the agent):
    position_sizing  bool  - may the strategy vary weights / total exposure / T0 size?
    allow_t0         bool  - are T+0 templates (T2, T4) allowed?
    costs            str   - configs/costs/<name>.yaml
    capital, base_n        - account size, number of T0 base stocks
    seq_models, custom_features

Templates
    T1  cross-sectional long book, multi-day hold, entries/exits after the decision bar
    T3  enter after the decision bar, exit at the next opening auction
    T2  base-position T0 overlay (intraday round trips, or overnight sell/buy-back)
    T4  T1 book used as the base for a T2 overlay
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass

from minute_agent.agent.space import MODELS, OPS, SEQ_MODELS, SpaceError, _check_value, _subset
from minute_agent.data.schema import bar_index_at

TEMPLATES = ["T1", "T2", "T3", "T4"]
T0_TEMPLATES = {"T2", "T4"}
DECISION_TIMES = ["09:45", "10:30", "11:20", "13:30", "14:00", "14:30", "14:50"]
T0_GRIDS = {
    "every_30": ["10:00", "10:30", "11:00", "13:30", "14:00", "14:30"],
    "every_15": ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15",
                 "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"],
}
T0_OVERNIGHT_DECISION = "14:50"
FLATTEN_AT = "14:45"
EXEC_STYLES = ["next_vwap", "twap5", "twap10"]
LABEL_NORMS = ["rank", "raw", "vol"]
NORMS = ["rank", "zscore", "none"]
RETRAIN = ["month", "quarter", "half", "year"]
WINDOW_YEARS = [1, 2, 3, 4]
MIN_PRICE = [0, 1, 2, 3, 5]
MIN_ADV_PCT = [0, 10, 20, 30, 50]

TARGETS = {
    "T1": {"kind": ["fwd_days"], "days": [1, 2, 5, 10]},
    "T3": {"kind": ["overnight"]},
    "T2": {"kind": ["intraday", "to_flatten", "overnight"], "minutes": [15, 30, 60]},
}

PORTFOLIO = {   # always open
    "top_frac": [0.01, 0.02, 0.05, 0.1],
    "buffer_frac": [0.0, 0.5, 1.0],
    "max_weight": [0.02, 0.05, 0.1],
    "rebalance_every": [1, 2, 5, 10],
    "hold_min": [1, 2, 5],
    "turnover_cap": [None, 0.2, 0.5],
    "stop_loss": [None, 0.05, 0.1],
}
PORTFOLIO_SIZING = {   # only with position_sizing on
    "weighting": ["equal", "score", "inv_vol"],
    "exposure_rule": ["full", "vol_target", "trend"],
    "target_vol": [0.1, 0.15, 0.2],
}
T0 = {
    "mode": ["intraday", "overnight"],
    "direction": ["sell_first", "buy_first", "both"],
    "size_frac": [0.25, 0.33, 0.5],
    "max_trips": [1, 2],
    "entry_q": [0.8, 0.9, 0.95],
    "hold_bars": [None, 15, 30, 60, 120],
    "take_profit": [None, 0.005, 0.01],
    "stop_loss": [None, 0.005, 0.01, 0.02],
    "signal_exit": [True, False],
}


@dataclass(frozen=True)
class Settings:
    position_sizing: bool = False
    allow_t0: bool = False
    costs: str = "default"
    capital: float = 1e8
    base_n: int = 300
    seq_models: bool = False
    custom_features: bool = False
    dataset: str = "cn"

    def templates(self) -> list[str]:
        return [t for t in TEMPLATES if self.allow_t0 or t not in T0_TEMPLATES]

    def as_dict(self) -> dict:
        return asdict(self)


def models(s: Settings) -> dict:
    return {**MODELS, **(SEQ_MODELS if s.seq_models else {})}


def _choice(where, v, dom):
    _check_value(where, v, "choice", dom)


def _check_features(feats, s: Settings):
    from minute_agent.agent.space import MAX_FEATURE_SPECS
    if not isinstance(feats, list) or not (1 <= len(feats) <= MAX_FEATURE_SPECS):
        raise SpaceError(f"features must be a list of 1..{MAX_FEATURE_SPECS} specs")
    seen = set()
    for f in feats:
        op = f.get("op")
        if op == "custom":
            if not s.custom_features:
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


def _check_target(t: dict, dom: dict, where: str):
    if not isinstance(t, dict):
        raise SpaceError(f"{where} must be an object")
    _choice(f"{where}.kind", t.get("kind"), dom["kind"])
    if t["kind"] == "fwd_days":
        _choice(f"{where}.days", t.get("days"), dom["days"])
    if t["kind"] == "intraday":
        _choice(f"{where}.minutes", t.get("minutes"), dom["minutes"])
    allowed = {"kind"} | ({"days"} if t["kind"] == "fwd_days" else set()) | ({"minutes"} if t["kind"] == "intraday" else set())
    if set(t) - allowed:
        raise SpaceError(f"{where}: unknown keys {sorted(set(t) - allowed)}")


def _check_block(name: str, blk: dict, dom: dict, required: bool = False):
    if not isinstance(blk, dict):
        raise SpaceError(f"{name} must be an object")
    extra = set(blk) - set(dom)
    if extra:
        raise SpaceError(f"{name}: keys {sorted(extra)} not allowed here")
    for k, v in blk.items():
        _choice(f"{name}.{k}", v, dom[k])
    if required:
        missing = set(dom) - set(blk)
        if missing:
            raise SpaceError(f"{name}: missing {sorted(missing)}")


def check(cfg: dict, s: Settings) -> dict:
    """Validate a v2 config against the action space and the session switches."""
    allowed = {"template", "decision_time", "t0_grid", "features", "target", "t0_target", "label_norm",
               "cross_section_norm", "model", "train", "portfolio", "t0", "execution", "universe_filter"}
    extra = set(cfg) - allowed - {"name", "seed"}
    if extra:
        raise SpaceError(f"keys outside the action space: {sorted(extra)}")
    c = copy.deepcopy(cfg)
    tpl = c.get("template")
    if tpl not in s.templates():
        raise SpaceError(f"template must be one of {s.templates()} in this session")
    _check_features(c.get("features"), s)
    _choice("label_norm", c.get("label_norm"), LABEL_NORMS)
    _choice("cross_section_norm", c.get("cross_section_norm"), NORMS)
    mdom = models(s)
    m = c.get("model") or {}
    if m.get("type") not in mdom:
        raise SpaceError(f"model.type must be one of {sorted(mdom)}")
    for p, v in (m.get("params") or {}).items():
        if p not in mdom[m["type"]]:
            raise SpaceError(f"model.params: unknown {p}")
        _choice(f"model.params.{p}", v, mdom[m["type"]][p])
    tr = c.get("train") or {}
    _check_block("train", tr, {"window_years": WINDOW_YEARS, "retrain_every": RETRAIN}, required=True)
    _choice("execution.style", (c.get("execution") or {}).get("style"), EXEC_STYLES)
    uf = c.get("universe_filter") or {}
    _check_block("universe_filter", uf, {"min_price": MIN_PRICE, "min_adv_pct": MIN_ADV_PCT})

    pdom = {**PORTFOLIO, **(PORTFOLIO_SIZING if s.position_sizing else {})}
    if tpl in ("T1", "T3", "T4"):
        _choice("decision_time", c.get("decision_time"), DECISION_TIMES)
        _check_target(c.get("target"), TARGETS["T3" if tpl == "T3" else "T1"], "target")
        _check_block("portfolio", c.get("portfolio") or {}, pdom)
        if "top_frac" not in (c.get("portfolio") or {}):
            raise SpaceError("portfolio.top_frac is required")
    if tpl in ("T2", "T4"):
        t0 = c.get("t0") or {}
        _check_block("t0", t0, T0, required=False)
        for k in ("mode", "direction", "size_frac", "entry_q"):
            if k not in t0:
                raise SpaceError(f"t0.{k} is required")
        tkey = "target" if tpl == "T2" else "t0_target"
        _check_target(c.get(tkey), TARGETS["T2"], tkey)
        tgt = c[tkey]
        if t0["mode"] == "overnight" and tgt.get("kind") != "overnight":
            raise SpaceError("t0.mode=overnight needs an overnight target")
        if t0["mode"] == "intraday":
            if tgt.get("kind") == "overnight":
                raise SpaceError("t0.mode=intraday needs an intraday or to_flatten target")
            _choice("t0_grid", c.get("t0_grid"), list(T0_GRIDS))
    if tpl == "T2" and ("portfolio" in c or "decision_time" in c):
        raise SpaceError("T2 has no portfolio/decision_time (the base is fixed by rule)")
    if tpl != "T4" and "t0_target" in c:
        raise SpaceError("t0_target is only for T4")
    if tpl in ("T1", "T3") and "t0" in c:
        raise SpaceError("t0 parameters only apply to T2/T4")
    return c


# ---------------------------------------------------------------------------


def decision_bars(cfg: dict) -> dict:
    """Decision bar indices per role: book (T1/T3/T4) and t0 (T2/T4)."""
    out = {}
    tpl = cfg["template"]
    if tpl in ("T1", "T3", "T4"):
        out["book"] = [bar_index_at("cn", cfg["decision_time"])]
    if tpl in ("T2", "T4"):
        if cfg["t0"]["mode"] == "overnight":
            out["t0"] = [bar_index_at("cn", T0_OVERNIGHT_DECISION)]
        else:
            out["t0"] = [bar_index_at("cn", h) for h in T0_GRIDS[cfg["t0_grid"]]]
    return out


def sample(rng: random.Random, s: Settings) -> dict:
    tpl = rng.choice(s.templates())
    ops = rng.sample(sorted(OPS), rng.randint(2, 7))
    feats = []
    for op in ops:
        spec = {"op": op}
        for p, (kind, dom) in OPS[op].items():
            spec[p] = rng.choice(dom) if kind == "choice" else _subset(rng, dom)
        feats.append(spec)
    mdom = models(s)
    mt = rng.choice(sorted(mdom))
    cfg = {
        "template": tpl, "features": feats,
        "label_norm": rng.choice(LABEL_NORMS), "cross_section_norm": rng.choice(NORMS),
        "model": {"type": mt, "params": {p: rng.choice(d) for p, d in mdom[mt].items()}},
        "train": {"window_years": rng.choice(WINDOW_YEARS), "retrain_every": rng.choice(RETRAIN)},
        "execution": {"style": rng.choice(EXEC_STYLES)},
        "universe_filter": {"min_price": rng.choice(MIN_PRICE), "min_adv_pct": rng.choice(MIN_ADV_PCT)},
    }
    pdom = {**PORTFOLIO, **(PORTFOLIO_SIZING if s.position_sizing else {})}
    if tpl in ("T1", "T3", "T4"):
        cfg["decision_time"] = rng.choice(DECISION_TIMES)
        cfg["target"] = ({"kind": "fwd_days", "days": rng.choice(TARGETS["T1"]["days"])} if tpl in ("T1", "T4")
                         else {"kind": "overnight"})
        cfg["portfolio"] = {k: rng.choice(v) for k, v in pdom.items()}
    if tpl in ("T2", "T4"):
        t0 = {k: rng.choice(v) for k, v in T0.items()}
        if t0["mode"] == "overnight":
            tgt = {"kind": "overnight"}
        else:
            kind = rng.choice(["intraday", "to_flatten"])
            tgt = {"kind": kind, **({"minutes": rng.choice(TARGETS["T2"]["minutes"])} if kind == "intraday" else {})}
            cfg["t0_grid"] = rng.choice(list(T0_GRIDS))
        cfg["t0"] = t0
        cfg["target" if tpl == "T2" else "t0_target"] = tgt
    return check(cfg, s)


def describe(s: Settings) -> str:
    from minute_agent.agent.space import describe as describe_v1
    feats = describe_v1(s.seq_models, s.custom_features).split("- cross_section_norm")[0]
    feats = feats.split("- features:", 1)[1]
    lines = [
        f"SESSION SWITCHES (fixed by the user): position_sizing={s.position_sizing}, allow_t0={s.allow_t0}",
        f"- template: one of {s.templates()}",
        "    T1: long book of the top names, entries/exits after the decision bar, multi-day holding",
        "    T3: buy after the decision bar, sell at the next opening auction",
    ]
    if s.allow_t0:
        lines += ["    T2: T0 overlay on a FIXED base portfolio (top base_n liquid stocks, equal weight, monthly);",
                  "        intraday round trips that are flattened from 14:45, or an overnight sell / buy-back",
                  "    T4: a T1 book that is also used as the base for a T0 overlay"]
    lines += [
        f"- decision_time (T1/T3/T4): one of {DECISION_TIMES}",
        f"- target: T1/T4 {{kind: fwd_days, days: one of {TARGETS['T1']['days']}}}; T3 {{kind: overnight}};",
        f"          T2 {{kind: intraday, minutes: one of {TARGETS['T2']['minutes']}}} | {{kind: to_flatten}} | {{kind: overnight}}",
        "  (T4 also needs t0_target, same choices as the T2 target). Returns start at the executable price after the",
        "  decision bar.",
        f"- label_norm: one of {LABEL_NORMS}; cross_section_norm: one of {NORMS}",
        "- features:" + feats.rstrip(),
        "- model: {type, params}: " + "; ".join(f"{k}: {list(v)}" for k, v in models(s).items()),
        f"- train: {{window_years: one of {WINDOW_YEARS}, retrain_every: one of {RETRAIN}}}",
        f"- execution: {{style: one of {EXEC_STYLES}}}  (fills after the decision bar; the engine is fixed)",
        f"- universe_filter: {{min_price: {MIN_PRICE}, min_adv_pct: {MIN_ADV_PCT}}}",
    ]
    pdom = {**PORTFOLIO, **(PORTFOLIO_SIZING if s.position_sizing else {})}
    lines.append("- portfolio (T1/T3/T4, top_frac required): " + ", ".join(f"{k} in {v}" for k, v in pdom.items()))
    if s.allow_t0:
        lines.append(f"- t0_grid (intraday T0): one of {list(T0_GRIDS)} = decision times {T0_GRIDS}")
        lines.append("- t0 (T2/T4; mode, direction, size_frac, entry_q required): " + ", ".join(f"{k} in {v}" for k, v in T0.items()))
        if s.position_sizing:
            lines.append("  with position_sizing on, T0 trade size also scales with signal strength")
    return "\n".join(lines)
