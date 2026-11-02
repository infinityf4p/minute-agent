"""Planner loop (LLM agent) and the random-search baseline.

Each round is stateless for the LLM: the prompt is rebuilt from the ledger
(best and recent attempts of this session) plus the agent's own memory notes,
so the context stays bounded however many rounds run. Every 10 rounds the
agent rewrites its memory notes (successful patterns, failure constraints).

Budget = number of EXECUTED experiments. Invalid or duplicate proposals are
rejected before execution, fed back to the agent, and don't use budget.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

from minute_agent.agent import space
from minute_agent.agent.executor import execute
from minute_agent.agent.llm import LLMClient, tool
from minute_agent.experiment import config as C
from minute_agent.experiment.ledger import Ledger
from minute_agent.paths import runs_root

FIXED = {"market": "cn", "eval_split": "valid", "seed": 0}
INTERRUPTED = "interrupted"


def compact(cfg: dict) -> str:
    """Short one-line description of a config for prompts and reports."""
    feats = []
    for f in cfg.get("features", []):
        if f["op"] == "custom":
            feats.append(f"custom:{f.get('name')}")
            continue
        ps = ",".join(f"{k}={v}" for k, v in f.items() if k != "op")
        feats.append(f"{f['op']}({ps})" if ps else f["op"])
    m = cfg.get("model", {})
    mp = ",".join(f"{k}={v}" for k, v in (m.get("params") or {}).items())
    uf = cfg.get("universe_filter") or {}
    tr = cfg.get("train") or {}
    return (f"target={cfg.get('target')} | univ(px>={uf.get('min_price', 0)},adv>={uf.get('min_adv_pct', 0)}%) | "
            f"feats=[{'; '.join(feats)}] | norm={cfg.get('cross_section_norm')}/{cfg.get('label_norm')} | "
            f"model={m.get('type')}({mp}) | train={tr.get('window_years')}y/{tr.get('retrain_every')}")


class Session:
    def __init__(self, kind: str, dataset: str, budget: int, name: str | None = None,
                 seq_models: bool = False, custom_features: bool = False, ledger_path: Path | None = None,
                 timeout_s: int = 1800):
        self.kind = kind
        self.dataset = dataset
        self.budget = budget
        self.seq_models = seq_models
        self.custom_features = custom_features
        self.name = name or f"{kind}-{datetime.now():%Y%m%d-%H%M%S}"
        self.dir = runs_root() / "sessions" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = ledger_path or (runs_root() / "ledger.sqlite")
        self.ledger = Ledger(self.ledger_path)
        self.timeout_s = timeout_s
        self.log = open(self.dir / "log.jsonl", "a", encoding="utf-8")
        meta = {"kind": kind, "dataset": dataset, "budget": budget, "seq_models": seq_models,
                "custom_features": custom_features}
        (self.dir / "session.json").write_text(json.dumps(meta, indent=1))
        # rows left 'running' by a killed process: mark them, they don't use budget
        for r in self.ledger.query("SELECT run_id FROM runs WHERE session=? AND status='running'", (self.name,)):
            self.ledger.fail(r["run_id"], INTERRUPTED)

    # -- ledger views ---------------------------------------------------------

    def runs(self) -> list[dict]:
        return self.ledger.query("SELECT * FROM runs WHERE session=? ORDER BY created_at", (self.name,))

    def executed(self) -> int:
        return sum(1 for r in self.runs() if not (r["status"] == "failed" and (r["error"] or "").startswith(INTERRUPTED)))

    def done_hashes(self) -> set[str]:
        return {r["config_hash"] for r in self.runs()}

    def emit(self, **kw):
        kw["t"] = datetime.now().isoformat(timespec="seconds")
        self.log.write(json.dumps(kw, default=str, ensure_ascii=False) + "\n")
        self.log.flush()

    def run_config(self, cfg: dict, rationale: str | None) -> dict:
        full = {**FIXED, "dataset": self.dataset, **cfg}
        res = execute(full, source=self.kind.split("-")[0], session=self.name, rationale=rationale,
                      ledger_path=self.ledger_path, workdir=self.dir / "runs", timeout_s=self.timeout_s)
        self.emit(event="result", run_id=res["run_id"], status=res["status"], seconds=res.get("seconds"),
                  objective=(res.get("metrics") or {}).get("objective"), error=res.get("error"))
        return res


# ---------------------------------------------------------------------------
# random search


def random_search(session: Session, seed: int = 0, verbose: bool = True) -> None:
    rng = random.Random(seed)
    seen = session.done_hashes()
    while session.executed() < session.budget:
        cfg = space.sample(rng, seq_models=session.seq_models)
        h = C.config_hash(C.validate({**FIXED, "dataset": session.dataset, **cfg}))
        if h in seen:
            continue
        seen.add(h)
        res = session.run_config(cfg, rationale="random sample")
        if verbose:
            print(f"[{session.name}] {session.executed()}/{session.budget} {res['status']} "
                  f"obj={(res.get('metrics') or {}).get('objective')} {res.get('seconds')}s", flush=True)


# ---------------------------------------------------------------------------
# LLM agent

SYSTEM = """You are a quantitative researcher running an automated search over machine-learning factor
pipelines built from Chinese A-share 1-minute bars.

Setting
- Each trading day, at the decision time 14:50, a model scores every stock in the universe using only bars
  up to 14:50 (plus daily history up to the previous day). Orders execute at today's close (closing auction).
- Targets: overnight = today's close -> tomorrow's open; next_cc = today's close -> tomorrow's close.
  A-shares are T+1 and cannot be shorted, so the portfolio is: equal-weight long the top decile of buyable
  stocks, benchmark = equal-weight universe. Costs: commission 2.5bp + slippage 2bp per side, plus sell-side
  stamp duty (10bp before 2023-08-28, 5bp after). overnight trades a full round trip every day; next_cc only
  pays for names entering/leaving the top decile.
- Evaluation: rolling retraining, scored on the validation period 2022-2023 only. The objective you must
  maximize is `excess_net_sharpe` = annualized Sharpe of (top decile - universe - costs). Also reported: IC,
  RankIC, ICIR, decile returns, turnover. A locked test period exists that you will never see; pipelines that
  merely fit validation noise will fail there, so prefer robust, economically sensible improvements.

How to work
- Each round, call `propose_experiment` with ONE config from the action space below and a short rationale.
- Use the history: build on what works, test one clear idea at a time, and keep some exploration.
- Exact duplicates of earlier configs are rejected. Runs that fail or time out still cost budget.
- Feature caching makes reusing feature specs cheap; brand-new feature params cost a few minutes once.
  Sequence models and month-by-month retraining are slow; use them deliberately.

ACTION SPACE
{space}
{custom}"""

CUSTOM_HELP = """
CUSTOM FEATURES (v2)
A feature spec {"op": "custom", "name": "<short_name>", "code": "<python source>"} adds your own feature.
The code must define `def feature(m, daily, d):` returning a polars DataFrame with columns symbol, date and
1..10 numeric columns. Only `import polars as pl`, `import numpy as np`, `import math` are allowed; no IO,
no dunder attributes, no eval/exec/getattr, no map_elements/apply.
- m: minute bars of many stocks and days, columns symbol, date, ts, bidx (0 = 09:30 auction, 1..120 =
  09:31..11:30, 121..240 = 13:01..15:00), open, high, low, close, volume (shares), amount (CNY), lr (1-minute
  log return within the day, null on bidx 0). Sorted by symbol, date, bidx.
- d: decision bar index (230 = 14:50). Your code MUST itself filter `pl.col("bidx") <= d`; it is tested on
  raw data and rejected if output depends on later bars.
- daily: daily bars (symbol, date, open, high, low, close, pre_close, volume, amount, turnover, adj_factor,
  up_limit, down_limit, is_suspended, is_st, float_mv, total_mv). It includes rows for the feature date and
  later, so only use daily rows from EARLIER dates (e.g. shift(1).over("symbol")) except pre_close,
  adj_factor, up_limit, down_limit, is_st, is_suspended of the same date (known pre-open).
- Every custom feature is checked automatically: values for day T must not change when all data after
  14:50 of day T is removed. Failing the check fails the run.
Example:
import polars as pl
def feature(m, daily, d):
    m = m.filter(pl.col("bidx") <= d)
    late = pl.col("bidx") > d - 20
    return m.group_by("symbol", "date").agg(
        (pl.col("volume").filter(late).sum() / pl.col("volume").sum()).alias("late_vol_share"))
"""

PROPOSE = tool("propose_experiment", "Run one experiment config on the validation period.", {
    "type": "object",
    "properties": {
        "config": {"type": "object", "description": "Config with keys target, universe_filter, features, "
                   "cross_section_norm, label_norm, model, train (see ACTION SPACE)."},
        "rationale": {"type": "string", "description": "Hypothesis being tested and why, 1-3 sentences."},
    },
    "required": ["config", "rationale"],
})
MEMORY = tool("update_memory", "Replace your research notes.", {
    "type": "object",
    "properties": {"notes": {"type": "string", "description": "Up to ~400 words: what works, what fails, "
                             "open hypotheses, constraints learned."}},
    "required": ["notes"],
})


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def history_text(session: Session, k_best: int = 8, k_recent: int = 12) -> str:
    rows = session.runs()
    if not rows:
        return "No experiments yet."

    def line(r):
        cfg = json.loads(r["config_json"])
        if r["status"] != "done":
            return f"- [{r['run_id'][-12:]}] FAILED: {compact(cfg)}\n    error: {(r['error'] or '')[-300:]}"
        m = json.loads(r["metrics_json"] or "{}")
        return (f"- [{r['run_id'][-12:]}] obj={_fmt(r['objective'])} netSR | IC={_fmt(r['ic'], 4)} "
                f"RankIC={_fmt(r['rank_ic'], 4)} ICIR={_fmt(r['icir'])} gross={_fmt(m.get('excess_gross_bp'), 1)}bp "
                f"net={_fmt(m.get('excess_net_bp'), 1)}bp turn={_fmt(m.get('top_turnover'), 2)} "
                f"t={_fmt(r['t_total'], 0)}s\n    {compact(cfg)}")
    done = [r for r in rows if r["status"] == "done" and r["objective"] is not None]
    best = sorted(done, key=lambda r: -r["objective"])[:k_best]
    recent = rows[-k_recent:]
    return ("BEST SO FAR (by objective):\n" + "\n".join(line(r) for r in best) +
            "\n\nMOST RECENT:\n" + "\n".join(line(r) for r in recent))


def agent_search(session: Session, client: LLMClient, verbose: bool = True, reflect_every: int = 10) -> None:
    notes_path = session.dir / "memory.md"
    system = SYSTEM.format(space=space.describe(session.seq_models, session.custom_features),
                           custom=CUSTOM_HELP if session.custom_features else "")
    last_detail = ""
    while session.executed() < session.budget:
        n = session.executed()
        if n and n % reflect_every == 0 and not (session.dir / f"reflect_{n}.done").exists():
            _reflect(session, client, system, notes_path)
            (session.dir / f"reflect_{n}.done").write_text("ok")
        notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(none yet)"
        user = (f"Budget: {n} of {session.budget} experiments used.\n\nYOUR RESEARCH NOTES:\n{notes}\n\n"
                f"{history_text(session)}\n\n{last_detail}\nPropose the next experiment.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        cfg, rationale = None, None
        for attempt in range(4):
            msg = client.chat(messages, tools=[PROPOSE])
            calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                messages.append({"role": "user", "content": "Call propose_experiment."})
                continue
            call = calls[0]
            err = None
            try:
                args = json.loads(call["function"]["arguments"])
                cand = space.check(args["config"], seq_models=session.seq_models,
                                   custom_features=session.custom_features)
                full = C.validate({**FIXED, "dataset": session.dataset, **cand})
                if C.config_hash(full) in session.done_hashes():
                    err = "This exact config was already run. Propose something different."
                else:
                    cfg, rationale = cand, args.get("rationale", "")
            except Exception as e:  # invalid JSON or outside the space
                err = f"Invalid proposal: {e}"
            for extra in calls[1:]:
                messages.append({"role": "tool", "tool_call_id": extra["id"], "content": "Ignored: one experiment per round."})
            if err is None:
                break
            session.emit(event="rejected", error=err)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": err})
        if cfg is None:
            session.emit(event="no_valid_proposal")
            time.sleep(5)
            continue
        session.emit(event="proposal", rationale=rationale, config=cfg)
        res = session.run_config(cfg, rationale)
        if res["status"] == "done":
            last_detail = ("LAST RESULT DETAIL:\n" + json.dumps({"rationale": rationale, "metrics": res["metrics"],
                                                                 "timings": res.get("timings")}, default=str))
        else:
            last_detail = f"LAST RUN FAILED: {res.get('error', '')[-800:]}"
        if verbose:
            print(f"[{session.name}] {session.executed()}/{session.budget} {res['status']} "
                  f"obj={(res.get('metrics') or {}).get('objective')} {res.get('seconds')}s | {rationale[:100]}",
                  flush=True)
    (session.dir / "llm_usage.json").write_text(json.dumps(client.usage, indent=1))


def _reflect(session: Session, client: LLMClient, system: str, notes_path: Path) -> None:
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(none yet)"
    user = (f"Time to consolidate. Current notes:\n{notes}\n\n{history_text(session, k_best=15, k_recent=20)}\n\n"
            "Rewrite your research notes with update_memory: successful patterns, failure constraints, "
            "and the most promising next directions. Be concrete (ops, params, models).")
    try:
        msg = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          tools=[MEMORY])
        call = (msg.get("tool_calls") or [None])[0]
        if call:
            text = json.loads(call["function"]["arguments"]).get("notes", "")
            if text:
                notes_path.write_text(text, encoding="utf-8")
                session.emit(event="memory", notes=text)
    except Exception as e:
        session.emit(event="memory_failed", error=repr(e))
