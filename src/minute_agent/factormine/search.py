"""Search arms: llm_code, llm_dsl, random. All share the evaluator, the library rules and the budget
(number of candidates submitted, including invalid and duplicate ones)."""

from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

from minute_agent.factormine import compute, config, data, dsl, evaluate, prompts
from minute_agent.factormine.library import Ledger, Library

ARMS = ("llm_code", "llm_dsl", "random")
BATCH = 4
LLM_MAX_TOKENS = 32000
LLM_REASONING = "medium"     # default (high) reasoning used >20k tokens per call on the code arm
# Scheduler prior from round-1 LLM sessions (mining-period admission counts per mechanism), down-weighted.
PRIOR_FILE = Path(__file__).with_name("mechanism_prior_r1.json")
PRIOR_WEIGHT = 0.5


class Session:
    def __init__(self, arm: str, seed: int, budget: int, cfg_name: str = "pilot", name: str | None = None):
        assert arm in ARMS, arm
        self.arm, self.seed, self.budget = arm, seed, budget
        self.cfg = config.get(cfg_name)
        self.periods = self.cfg["mine"]
        self.name = name or f"{cfg_name}-{arm}-s{seed}"
        self.dir = config.runs_dir() / "sessions" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"arm": arm, "seed": seed, "budget": budget, "cfg": cfg_name, "fingerprint": config.fingerprint(self.cfg)}
        mp = self.dir / "session.json"
        if mp.exists():
            old = json.loads(mp.read_text())
            if old["fingerprint"] != meta["fingerprint"] or old["arm"] != arm:
                raise RuntimeError(f"session {self.name} exists with other settings: {old}")
        mp.write_text(json.dumps(meta, indent=1))
        self.ledger = Ledger(config.runs_dir() / "ledger.sqlite")
        self.lib = Library(self.dir / "library")
        self.ref = evaluate.reference_frame(self.cfg, self.periods)
        if config.THRESHOLDS["seed_reference"] and not self.lib.members:
            self._seed_library()

    def _seed_library(self) -> None:
        for name, node in evaluate.REFERENCE_NODES.items():
            spec = {"name": name, "mechanism": "reference", "hypothesis": "known intraday factor",
                    "intraday": {"a": node}, "combine": {"var": "a"}}
            m, o = evaluate.metrics(evaluate.factor_frame(self.cfg, spec, self.periods), self.periods)
            self.lib.add(name, spec, m, o, 0, None, fixed=True)

    def rows(self):
        return self.ledger.rows(self.name)

    def n(self) -> int:
        return len(self.rows())

    def emit(self, **kw) -> None:
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}, default=str) + "\n")

    def record(self, spec: dict | None, mechanism: str, status: str, reason: str, m: dict | None, secs: float,
               key: str | None = None) -> dict:
        n = self.n() + 1
        name = (spec or {}).get("name", "?") if isinstance(spec, dict) else "?"
        self.ledger.add(session=self.name, arm=self.arm, seed=self.seed, n=n, mechanism=mechanism, name=str(name),
                        spec_key=key, spec_json=json.dumps(spec, default=str) if spec is not None else None,
                        status=status, reason=reason[:1000], metrics_json=json.dumps(m, default=str) if m else None,
                        secs=round(secs, 2))
        out = {"n": n, "name": name, "mechanism": mechanism, "status": status, "reason": reason, "metrics": m or {}}
        self.emit(event="candidate", **{k: v for k, v in out.items() if k != "metrics"},
                  res_ic=(m or {}).get("res_ic"), res_t=(m or {}).get("res_t"))
        print(f"[{self.name}] {n}/{self.budget} {mechanism} {name}: {status} {reason[:120]} "
              f"res_t={(m or {}).get('res_t', float('nan')):+.1f}", flush=True)
        return out

    # ------------------------------------------------------------------ evaluation of a batch
    def evaluate_batch(self, raws: list, mechanism: str) -> list[dict]:
        out, valid = [], []
        seen = {r["spec_key"] for r in self.rows() if r["spec_key"]}
        for raw in raws:
            t0 = time.time()
            try:
                if isinstance(raw, dict):
                    if self.arm == "llm_code" and (raw.get("intraday") or not raw.get("code")):
                        raise evaluate.SpecError("this arm writes the intraday stage as code (the code field)")
                    if self.arm != "llm_code" and raw.get("code"):
                        raise evaluate.SpecError("this arm writes the intraday stage in the minute DSL, not code")
                spec = evaluate.validate(raw, self.cfg)
                spec["mechanism"] = mechanism
                key = evaluate.spec_key(spec)
                if key in seen:
                    out.append(self.record(spec, mechanism, "duplicate", "same factor already evaluated", None,
                                           time.time() - t0, key))
                    continue
                seen.add(key)
                if spec.get("code"):
                    compute.check_code_leakage(self.cfg, spec["code"])
                valid.append((spec, key, time.time() - t0))
            except Exception as e:
                out.append(self.record(raw if isinstance(raw, dict) else None, mechanism, "invalid",
                                       f"{type(e).__name__}: {e}", None, time.time() - t0))
        if not valid:
            return out
        t0 = time.time()
        start, end = config.span(self.periods)
        units = [u for s, _, _ in valid for u in evaluate.units_of(s)[0]]
        vals = compute.compute(self.cfg, units, data.shift_months(start, self.cfg["warmup_months"]), end)
        share = (time.time() - t0) / len(valid)
        for spec, key, pre in valid:
            t1 = time.time()
            try:
                fr = evaluate.factor_frame(self.cfg, spec, self.periods, vals)
                m, o = evaluate.metrics(fr, self.periods)
                corrs = evaluate.corr_with(o, self.lib.values) if "error" not in m else {}
                refc = evaluate.corr_with(o, self.ref) if "error" not in m else {}
                if corrs:
                    k = max(corrs, key=lambda c: abs(corrs[c]))
                    m["max_lib_corr"], m["max_lib_member"] = corrs[k], k
                if refc:
                    k = max(refc, key=lambda c: abs(refc[c]))
                    m["max_ref_corr"], m["max_ref"] = refc[k], k
                status, replaced, reason = self.lib.decide(m, corrs)
                if status in ("admitted", "replaced"):
                    mid = f"{spec['name'][:24]}_{self.n() + 1}"
                    self.lib.add(mid, spec, m, o, self.n() + 1, replaced)
                    reason = f"{reason}; id {mid}"
                out.append(self.record(spec, mechanism, status, reason, m, pre + share + time.time() - t1, key))
            except Exception as e:
                out.append(self.record(spec, mechanism, "failed", f"{type(e).__name__}: {e}"[:800], None,
                                       pre + share + time.time() - t1, key))
        return out


# ---------------------------------------------------------------------- random arm

def run_random(sess: Session) -> None:
    rng = random.Random(sess.seed)
    for _ in range(sess.n()):          # resume: advance the generator past evaluated candidates
        dsl.random_factor(rng, data.DAILY_VARS)
    while sess.n() < sess.budget:
        k = min(BATCH, sess.budget - sess.n())
        sess.evaluate_batch([dsl.random_factor(rng, data.DAILY_VARS) for _ in range(k)], "random")


# ---------------------------------------------------------------------- LLM arms

def mechanism_stats(sess: Session) -> dict[str, tuple[int, int]]:
    st = {m: [0, 0] for m in prompts.MECHANISMS}
    for r in sess.rows():
        if r["mechanism"] in st:
            st[r["mechanism"]][0 if r["status"] in ("admitted", "replaced") else 1] += 1
    return {k: (a, b) for k, (a, b) in st.items()}


def mechanism_prior() -> dict[str, tuple[float, float]]:
    if not PRIOR_FILE.exists():
        return {}
    c = json.loads(PRIOR_FILE.read_text())["counts"]
    return {m: (PRIOR_WEIGHT * v["admitted"], PRIOR_WEIGHT * v["other"]) for m, v in c.items()}


def pick_mechanism(sess: Session, rng: random.Random) -> str:
    st = mechanism_stats(sess)
    pr = mechanism_prior()
    draws = {m: rng.betavariate(1 + s + pr.get(m, (0, 0))[0], 1 + f + pr.get(m, (0, 0))[1])
             for m, (s, f) in st.items()}
    return max(draws, key=draws.get)


def _fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def _spec_text(spec: dict, limit: int = 700) -> str:
    core = {"code": spec["code"]} if spec.get("code") else {"intraday": spec.get("intraday")}
    s = json.dumps({**core, "combine": spec.get("combine")}, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def user_prompt(sess: Session, mech: str, feedback: str) -> str:
    rows = sess.rows()
    found = sess.lib.found()
    lib_lines = [f"  - {x['id']} [{x['spec'].get('mechanism')}] res_ic={_fmt(x['metrics']['res_ic'])} "
                 f"t={_fmt(x['metrics']['res_t'], 1)}: {x['spec'].get('hypothesis', '')[:140]}\n"
                 f"    {_spec_text(x['spec'], 400)}" for x in found[-25:]]
    recent = []
    for r in rows[-12:]:
        m = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        recent.append(f"  - #{r['n']} {r['name']} [{r['mechanism']}] {r['status']}: {r['reason'][:160]} | "
                      f"raw_ic={_fmt(m.get('ic'))} res_ic={_fmt(m.get('res_ic'))} t={_fmt(m.get('res_t'), 1)} folds="
                      f"{[round(x, 4) for x in m.get('res_ic_folds', [])]} max_corr={_fmt(m.get('max_lib_corr'), 2)}"
                      f" ({m.get('max_lib_member', '-')})")
    near = []
    for r in rows:
        if r["mechanism"] == mech and r["status"] in ("rejected_weak", "rejected_redundant") and r["metrics_json"]:
            m = json.loads(r["metrics_json"])
            if m.get("res_t") is not None and np.isfinite(m["res_t"]):
                near.append((abs(m["res_t"]), r, m))
    near = sorted(near, key=lambda x: -x[0])[:2]
    near_lines = [f"  - {r['name']} ({r['status']}: {r['reason'][:120]}; t={_fmt(m['res_t'], 1)})\n"
                  f"    {_spec_text(json.loads(r['spec_json']))}" for _, r, m in near]
    redundant = [f"  - {r['name']}: {r['reason'][:120]}" for r in rows if r["status"] == "rejected_redundant"][-8:]
    k = min(BATCH, sess.budget - sess.n())
    return "\n".join([
        f"Budget: {sess.n()} of {sess.budget} candidates used.",
        f"ASSIGNED MECHANISM for this round: {mech} - {prompts.MECHANISMS[mech]}",
        "",
        f"LIBRARY: factors found so far ({len(found)}; the fixed seed factors are listed in the system prompt):",
        *(lib_lines or ["  (none yet)"]),
        "",
        "RECENT RESULTS:", *(recent or ["  (none yet)"]),
        "",
        "REJECTED AS REDUNDANT (avoid these directions):", *(redundant or ["  (none)"]),
        "",
        f"NEAR MISSES for {mech} (candidates to improve by changing one element):", *(near_lines or ["  (none)"]),
        "",
        feedback,
        f"Propose exactly {k} factors for the mechanism {mech}. If there are near misses, make one of them a targeted "
        "variation of a near miss; the others should be new ideas within the mechanism that differ from the "
        "library and from each other.",
    ])


def run_llm(sess: Session, client) -> None:
    from minute_agent.agent.llm import LLMClient  # noqa: F401  (client is an LLMClient)
    rng = random.Random(1000 + sess.seed + sess.n())
    seeds = [{"id": x["id"], "desc": json.dumps(x["spec"]["intraday"]["a"])} for x in sess.lib.members if x.get("fixed")]
    system = prompts.system_prompt(sess.arm, BATCH, seeds, prompts.CODE_HELP)
    tool = prompts.tool_schema(sess.arm, BATCH)
    feedback = ""
    while sess.n() < sess.budget:
        mech = pick_mechanism(sess, rng)
        user = user_prompt(sess, mech, feedback)
        factors, note = None, ""
        for attempt in range(3):
            msg = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user + note}],
                              tools=[tool])
            calls = msg.get("tool_calls") or []
            try:
                args = json.loads(calls[0]["function"]["arguments"])
                factors = args["factors"]
                if not isinstance(factors, list) or not factors:
                    raise ValueError("factors must be a non-empty list")
                break
            except Exception as e:
                factors = None
                note = f"\n\n(Your previous reply was not a valid propose_factors call: {e}. Call propose_factors.)"
        if factors is None:
            sess.record(None, mech, "invalid", f"no valid propose_factors call after 3 attempts: {note[-300:]}",
                        None, 0.0)
            continue
        k = min(BATCH, sess.budget - sess.n())
        res = sess.evaluate_batch(factors[:k], mech)
        feedback = "LAST BATCH:\n" + "\n".join(
            f"  - {r['name']}: {r['status']} - {r['reason'][:300]} (raw_ic={_fmt(r['metrics'].get('ic'))}, "
            f"res_ic={_fmt(r['metrics'].get('res_ic'))})" for r in res) + "\n"
        sess.emit(event="round", mechanism=mech, tokens=client.usage)


def run(arm: str, seed: int, budget: int, cfg_name: str = "pilot", tag: str = "") -> Session:
    sess = Session(arm, seed, budget, cfg_name, name=f"{cfg_name}-{arm}{tag}-s{seed}")
    sess.emit(event="start", n=sess.n())
    try:
        if arm == "random":
            run_random(sess)
        else:
            from minute_agent.agent.llm import LLMClient
            client = LLMClient(log_path=sess.dir / "llm_calls.jsonl", temperature=0.7 + 0.1 * (seed % 3),
                               max_tokens=LLM_MAX_TOKENS, extra_body={"reasoning_effort": LLM_REASONING})
            run_llm(sess, client)
    except Exception:
        sess.emit(event="crash", error=traceback.format_exc()[-2000:])
        raise
    sess.emit(event="done", n=sess.n(), found=len(sess.lib.found()))
    return sess
