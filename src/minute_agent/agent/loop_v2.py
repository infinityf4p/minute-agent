"""v2 search over whole trading workflows: LLM agent and random-search arms.

Each proposal is evaluated in two levels, identical for every arm:
  1. screen: one fold (train ... 2020, evaluate 2021)
  2. folds:  evaluate 2020 and 2021 (only if the screen result is in the top 30% of
     this session's screens for the same template, or among its first 3)
Both levels count as trials (budget = number of evaluations). The agent only ever
sees training-period results; validation and test are run by the evaluator later.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from minute_agent import split
from minute_agent.agent.llm import LLMClient, tool
from minute_agent.agent.space import SpaceError
from minute_agent.backtest.costs import CostModel
from minute_agent.experiment.ledger import Ledger
from minute_agent.paths import runs_root
from minute_agent.pipeline import spec
from minute_agent.pipeline.run import config_hash

INTERRUPTED = "interrupted"
PROMOTE_Q = 0.7
PROMOTE_MIN = 3


class SessionV2:
    def __init__(self, arm: str, budget: int, settings: spec.Settings, name: str | None = None,
                 seed: int = 0, ledger_path: Path | None = None, timeout_s: int = 2400):
        self.arm, self.budget, self.settings, self.seed = arm, budget, settings, seed
        self.name = name or f"v2-{arm}-s{seed}-{datetime.now():%Y%m%d-%H%M%S}"
        self.dir = runs_root() / "sessions" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = ledger_path or runs_root() / "ledger.sqlite"
        self.ledger = Ledger(self.ledger_path)
        self.timeout_s = timeout_s
        (self.dir / "session.json").write_text(json.dumps(
            {"arm": arm, "budget": budget, "seed": seed, "settings": settings.as_dict()}, indent=1))
        for r in self.ledger.query("SELECT run_id FROM runs WHERE session=? AND status='running'", (self.name,)):
            self.ledger.fail(r["run_id"], INTERRUPTED)
        self.log = open(self.dir / "log.jsonl", "a", encoding="utf-8")

    def rows(self) -> list[dict]:
        out = []
        for r in self.ledger.query("SELECT * FROM runs WHERE session=? ORDER BY created_at", (self.name,)):
            if r["status"] == "failed" and (r["error"] or "").startswith(INTERRUPTED):
                continue
            r["record"] = json.loads(r["config_json"])
            r["metrics"] = json.loads(r["metrics_json"] or "{}")
            out.append(r)
        return out

    def used(self) -> int:
        return len(self.rows())

    def emit(self, **kw):
        kw["t"] = datetime.now().isoformat(timespec="seconds")
        self.log.write(json.dumps(kw, default=str, ensure_ascii=False) + "\n")
        self.log.flush()

    # ---- evaluation (subprocess, unlock variable stripped)

    def evaluate(self, cfg: dict, phase: str, rationale: str | None) -> dict:
        chash = config_hash(cfg, self.settings)
        run_id = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{chash}"
        wd = self.dir / "runs"
        wd.mkdir(exist_ok=True)
        paths = {k: wd / f"{run_id}.{k}.json" for k in ("config", "result", "settings")}
        paths["config"].write_text(json.dumps(cfg, indent=1))
        paths["settings"].write_text(json.dumps(self.settings.as_dict()))
        cmd = [sys.executable, "-m", "minute_agent.pipeline.execute", str(paths["config"]), str(paths["result"]),
               "--settings", str(paths["settings"]), "--phase", phase, "--run-id", run_id, "--source", self.arm,
               "--session", self.name, "--ledger", str(self.ledger_path)]
        if rationale:
            rp = wd / f"{run_id}.rationale.txt"
            rp.write_text(rationale, encoding="utf-8")
            cmd += ["--rationale-file", str(rp)]
        env = {k: v for k, v in os.environ.items() if k != split.UNLOCK_ENV}
        env["PYTHONIOENCODING"] = "utf-8"
        t0 = time.time()
        err = None
        try:
            p = subprocess.run(cmd, env=env, timeout=self.timeout_s, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if p.returncode:
                err = p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            err = f"timeout after {self.timeout_s}s"
        res = json.loads(paths["result"].read_text()) if paths["result"].exists() else {"status": "failed", "error": err}
        if res.get("status") != "done":
            try:
                self.ledger.fail(run_id, res.get("error") or err or "failed", {"total": time.time() - t0})
            except Exception:
                pass
        res.update({"run_id": run_id, "phase": phase, "seconds": round(time.time() - t0, 1)})
        self.emit(event="eval", phase=phase, run_id=run_id, status=res["status"],
                  objective=(res.get("metrics") or {}).get("objective"), error=res.get("error"))
        return res

    def promote(self, template: str, screen_obj: float) -> bool:
        prev = [r["objective"] for r in self.rows() if r["eval_split"] == "screen" and r["status"] == "done"
                and r["record"].get("template") == template and r["objective"] is not None]
        prev = prev[:-1]  # exclude the run just finished
        if len(prev) < PROMOTE_MIN:
            return True
        return screen_obj >= float(np.quantile(prev, PROMOTE_Q))

    def run_proposal(self, cfg: dict, rationale: str | None) -> dict:
        out = {"screen": self.evaluate(cfg, "screen", rationale)}
        sc = out["screen"]
        if sc["status"] == "done" and self.used() < self.budget and np.isfinite(sc["metrics"].get("objective", np.nan)) \
                and self.promote(cfg["template"], sc["metrics"]["objective"]):
            out["folds"] = self.evaluate(cfg, "folds", rationale)
        return out

    def seen_hashes(self) -> set[str]:
        return {r["config_hash"] for r in self.rows()}


# ---------------------------------------------------------------------------
# random arm


def random_search(sess: SessionV2, verbose: bool = True):
    rng = random.Random(sess.seed)
    seen = sess.seen_hashes()
    while sess.used() < sess.budget:
        cfg = spec.sample(rng, sess.settings)
        h = config_hash(cfg, sess.settings)
        if h in seen:
            continue
        seen.add(h)
        r = sess.run_proposal(cfg, "random sample")
        if verbose:
            print(f"[{sess.name}] {sess.used()}/{sess.budget} {cfg['template']} screen="
                  f"{(r['screen'].get('metrics') or {}).get('objective')} folds="
                  f"{((r.get('folds') or {}).get('metrics') or {}).get('objective')}", flush=True)


# ---------------------------------------------------------------------------
# LLM arm

SYSTEM = """You are a quantitative researcher running an automated search over complete trading workflows for
Chinese A-shares, built from 1-minute bars (241 bars/day incl. opening/closing auctions) and daily bars.

What you choose (the action space below): the strategy template, decision time(s), features, prediction target,
model, training window, portfolio / T0 parameters and an execution style. What is fixed and you cannot change:
the backtest engine (fills after the decision bar at the chosen style's price, a participation cap of the bar
volume, no fills when a stock is locked at its price limit, T+1 for newly bought shares, forced T0 flatten from
14:45 with any unfilled leg settled at the next opening auction), the data, the costs, and the evaluation.

Costs (set by the user): {costs}
Session switches (set by the user): position_sizing={ps}, allow_t0={t0}.

Objectives (higher is better):
- T1 / T3 / T4: annualized Sharpe of the hedged net excess return (portfolio minus exposure x equal-weight
  universe over the same intervals, after costs).
- T2: mean daily T0 overlay return, net of costs, in basis points of the base portfolio value per day.
You also see the result at 2x costs and before costs; a strategy that only works before costs is not useful.

Evaluation you will see: every proposal is first screened on one fold (train on data up to 2020, evaluate 2021).
If its screen objective is in the top 30% of your screens for that template it is also evaluated on two folds
(evaluate 2020 and 2021). Both evaluations count against your budget. You never see later years; the final
choice among your best pipelines is made on unseen data, so prefer robust, economically sensible ideas over
fitting noise.

Each round call `propose_workflow` with one config and a short rationale. Use the history; test one clear idea at a
time; keep some exploration across templates. Exact duplicates are rejected. New (op, decision time) combinations
and new execution styles cost extra compute the first time.

ACTION SPACE
{space}"""

ROBUST = """

ROBUSTNESS RULES FOR THIS SESSION
- A scheduler assigns the template for each round, from how often each template has produced configs that are
  positive in BOTH folds (Thompson sampling, every template tried at least {min_n} times). You must propose a
  config of the assigned template.
- Judge configs by min_fold = the worse of the two fold objectives, not by the screen result or the fold mean: a
  single good year is usually noise and does not survive on unseen years. Prefer changes that make a config
  positive in both years and still positive at 2x costs."""

SCHED_MIN = 2   # proposals per template before the scheduler samples


def template_stats(sess: SessionV2) -> dict[str, list[int]]:
    """Per template: [successes, failures]. Success = promoted to folds and positive in both folds."""
    st = {t: [0, 0] for t in sess.settings.templates()}
    rows = sess.rows()
    folds = {r["config_hash"]: r for r in rows if r["eval_split"] == "folds"}
    for r in rows:
        if r["eval_split"] != "screen" or r["record"]["cfg"].get("template") not in st:
            continue
        f = folds.get(r["config_hash"])
        ok = False
        if f is not None and f["status"] == "done":
            a, b = f["metrics"].get("fold1_objective"), f["metrics"].get("fold2_objective")
            ok = a is not None and b is not None and min(a, b) > 0
        st[r["record"]["cfg"]["template"]][0 if ok else 1] += 1
    return st


def pick_template(sess: SessionV2, rng: random.Random) -> tuple[str, dict]:
    st = template_stats(sess)
    under = sorted(t for t, (s, f) in st.items() if s + f < SCHED_MIN)
    if under:
        return rng.choice(under), st
    return max(st, key=lambda t: rng.betavariate(1 + st[t][0], 1 + st[t][1])), st


PROPOSE = tool("propose_workflow", "Evaluate one workflow config.", {
    "type": "object",
    "properties": {"config": {"type": "object"}, "rationale": {"type": "string"}},
    "required": ["config", "rationale"],
})
MEMORY = tool("update_memory", "Replace your research notes.", {
    "type": "object", "properties": {"notes": {"type": "string"}}, "required": ["notes"],
})


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _short(cfg: dict) -> str:
    feats = ",".join(f["op"] if f["op"] != "custom" else f"custom:{f.get('name')}" for f in cfg.get("features", []))
    parts = [cfg["template"]]
    if "decision_time" in cfg:
        parts.append(cfg["decision_time"])
    if "t0_grid" in cfg:
        parts.append(cfg["t0_grid"])
    parts.append(f"tgt={json.dumps(cfg.get('target'))}")
    if "t0_target" in cfg:
        parts.append(f"t0tgt={json.dumps(cfg['t0_target'])}")
    parts.append(f"model={cfg['model']['type']}{json.dumps(cfg['model'].get('params', {}))}")
    parts.append(f"train={cfg['train']['window_years']}y/{cfg['train']['retrain_every']}")
    if "portfolio" in cfg:
        parts.append(f"port={json.dumps(cfg['portfolio'])}")
    if "t0" in cfg:
        parts.append(f"t0={json.dumps(cfg['t0'])}")
    parts.append(f"exec={cfg['execution']['style']} univ={json.dumps(cfg.get('universe_filter'))}")
    parts.append(f"feats=[{feats}]")
    return " | ".join(parts)


def _min_fold(r) -> float | None:
    a, b = r["metrics"].get("fold1_objective"), r["metrics"].get("fold2_objective")
    return None if a is None or b is None else min(a, b)


def history_text(sess: SessionV2, k_best=8, k_recent=12, robust: bool = False) -> str:
    rows = sess.rows()
    if not rows:
        return "No evaluations yet."

    def line(r):
        cfg = r["record"]["cfg"]
        if r["status"] != "done":
            return f"- [{r['eval_split']}] FAILED {_short(cfg)}\n    error: {(r['error'] or '')[-300:]}"
        m = r["metrics"]
        return (f"- [{r['eval_split']}] obj={_fmt(m.get('objective'))} (x2 cost {_fmt(m.get('x2_objective'))}, gross "
                f"{_fmt(m.get('gross_objective'))}) folds=({_fmt(m.get('fold1_objective'))},{_fmt(m.get('fold2_objective'))}) "
                + (f"min_fold={_fmt(_min_fold(r))} " if robust and r["eval_split"] == "folds" else "") +
                f"RankIC={_fmt(m.get('rank_ic'), 4)} {_fmt(m.get('mean_bp_per_day'), 2)}bp/d turnover={_fmt(m.get('turnover'), 2)} "
                f"t={_fmt(r['t_total'], 0)}s\n    {_short(cfg)}")
    folds = [r for r in rows if r["eval_split"] == "folds" and r["status"] == "done" and r["objective"] is not None]
    by_t = {}
    for r in folds:
        by_t.setdefault(r["record"]["cfg"]["template"], []).append(r)
    best = []
    for t, rs in by_t.items():
        mf = (lambda r: -(_min_fold(r) if _min_fold(r) is not None else -1e9))
        best += sorted(rs, key=mf if robust else (lambda r: -r["objective"]))[:max(2, k_best // max(len(by_t), 1))]
    head = "BEST TWO-FOLD RESULTS PER TEMPLATE (by min_fold):\n" if robust else "BEST TWO-FOLD RESULTS PER TEMPLATE:\n"
    return (head + ("\n".join(line(r) for r in best) or "(none yet)")
            + "\n\nMOST RECENT EVALUATIONS:\n" + "\n".join(line(r) for r in rows[-k_recent:]))


def agent_search(sess: SessionV2, client: LLMClient, verbose: bool = True, reflect_every: int = 10,
                 scheduler: str | None = None):
    """scheduler=None: the agent picks templates freely (P4 arm). "ts": a Thompson-sampling scheduler assigns the
    template each round and the agent is told to judge configs by the worse fold (robust arm)."""
    costs = CostModel.load(sess.settings.costs)
    system = SYSTEM.format(costs=json.dumps(costs.raw, default=str), ps=sess.settings.position_sizing,
                           t0=sess.settings.allow_t0, space=spec.describe(sess.settings))
    robust = scheduler == "ts"
    if robust:
        system += ROBUST.format(min_n=SCHED_MIN)
    rng = random.Random(1000 + sess.seed)
    notes_path = sess.dir / "memory.md"
    last = ""
    rounds = 0
    while sess.used() < sess.budget:
        rounds += 1
        if rounds > 1 and rounds % reflect_every == 1:
            _reflect(sess, client, system, notes_path, robust)
        notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(none yet)"
        want, sched = None, ""
        if robust:
            want, st = pick_template(sess, rng)
            sess.emit(event="schedule", template=want, stats=st)
            sched = ("SCHEDULER: template successes/failures so far (success = positive in both folds): "
                     + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in st.items())
                     + f".\nThis round you must propose a {want} workflow.\n")
        user = (f"Budget: {sess.used()} of {sess.budget} evaluations used.\n\nYOUR NOTES:\n{notes}\n\n"
                f"{history_text(sess, robust=robust)}\n\n{last}\n{sched}Propose the next workflow.")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        cfg = rationale = None
        seen = sess.seen_hashes()
        for _ in range(4):
            msg = client.chat(msgs, tools=[PROPOSE])
            calls = msg.get("tool_calls") or []
            msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                msgs.append({"role": "user", "content": "Call propose_workflow."})
                continue
            call, err = calls[0], None
            try:
                args = json.loads(call["function"]["arguments"])
                cand = spec.check(args["config"], sess.settings)
                if want and cand["template"] != want:
                    err = f"This round the template must be {want}."
                elif config_hash(cand, sess.settings) in seen:
                    err = "This exact config was already evaluated. Propose something different."
                else:
                    cfg, rationale = cand, args.get("rationale", "")
            except (SpaceError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as e:
                err = f"Invalid proposal: {e}"
            for extra in calls[1:]:
                msgs.append({"role": "tool", "tool_call_id": extra["id"], "content": "Ignored: one proposal per round."})
            if err is None:
                break
            sess.emit(event="rejected", error=err)
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": err})
        if cfg is None:
            sess.emit(event="no_valid_proposal")
            continue
        sess.emit(event="proposal", rationale=rationale, config=cfg)
        r = sess.run_proposal(cfg, rationale)
        last = "LAST PROPOSAL RESULT:\n" + json.dumps({k: {kk: r[k].get(kk) for kk in ("status", "metrics", "error")}
                                                        for k in ("screen", "folds") if k in r}, default=str)[:3000]
        if verbose:
            print(f"[{sess.name}] {sess.used()}/{sess.budget} {cfg['template']} screen="
                  f"{(r['screen'].get('metrics') or {}).get('objective')} folds="
                  f"{((r.get('folds') or {}).get('metrics') or {}).get('objective')} | {rationale[:90]}", flush=True)
    (sess.dir / "llm_usage.json").write_text(json.dumps(client.usage, indent=1))


def _reflect(sess, client, system, notes_path, robust: bool = False):
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "(none yet)"
    user = (f"Consolidate. Current notes:\n{notes}\n\n{history_text(sess, 16, 20, robust)}\n\nRewrite your notes with "
            "update_memory (max ~400 words): what works per template, what fails, constraints, next directions.")
    try:
        msg = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], tools=[MEMORY])
        call = (msg.get("tool_calls") or [None])[0]
        if call:
            text = json.loads(call["function"]["arguments"]).get("notes", "")
            if text:
                notes_path.write_text(text, encoding="utf-8")
                sess.emit(event="memory", notes=text)
    except Exception as e:
        sess.emit(event="memory_failed", error=repr(e))
