"""Summary of factor-mining sessions: search efficiency (mining period) and validation (2022-2023).

    conda run --no-capture-output -n minute-agent python scripts/factormine/fm_report.py runs/factormine/validation_pilot.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import numpy as np
import polars as pl

from minute_agent.factormine import config
from minute_agent.factormine.library import Library

KEYS = ["symbol", "date"]


def mean_abs_pair_corr(values: pl.DataFrame, ids: list[str]) -> float:
    if len(ids) < 2:
        return float("nan")
    pairs = list(combinations(ids, 2))
    per = values.group_by("date").agg([pl.corr(a, b).alias(f"{a}|{b}") for a, b in pairs])
    return float(np.nanmean([abs(per[f"{a}|{b}"].drop_nans().drop_nulls().mean() or 0) for a, b in pairs]))


def session_row(name: str, val: dict) -> dict:
    sd = config.runs_dir() / "sessions" / name
    c = sqlite3.connect(config.runs_dir() / "ledger.sqlite")
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT * FROM candidates WHERE session=? ORDER BY id", (name,)).fetchall()
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    lib = Library(sd / "library")
    found = lib.found()
    tokens, llm_s = 0, 0.0
    lp = sd / "llm_calls.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            x = json.loads(line)
            tokens += x["usage"].get("total_tokens", 0)
            llm_s += x["secs"]
    t0, t1 = (datetime.fromisoformat(rows[0]["created_at"]), datetime.fromisoformat(rows[-1]["created_at"]))
    v = val.get(name, {})
    ov = [m["valid"]["res_ic"] * np.sign(m["mine"]["res_ic"]) for m in v.get("members", [])
          if m["valid"].get("res_ic") is not None]
    ot = [m["valid"]["res_t"] * np.sign(m["mine"]["res_ic"]) for m in v.get("members", [])
          if m["valid"].get("res_t") is not None]
    vs, vf, va = v.get("valid_seeds") or {}, v.get("valid_found") or {}, v.get("valid_all") or {}
    return {
        "session": name, "arm": json.loads((sd / "session.json").read_text())["arm"], "n": len(rows),
        "found": len(found), "admit_events": st["admitted"] + st["replaced"], "replaced": st["replaced"],
        "weak": st["rejected_weak"], "redundant": st["rejected_redundant"],
        "invalid": st["invalid"] + st["failed"] + st["rejected_invalid"], "dup": st["duplicate"],
        "tokens_k": tokens / 1000, "llm_min": llm_s / 60, "wall_min": (t1 - t0).total_seconds() / 60,
        "mechs": len({x["spec"].get("mechanism") for x in found}),
        "pair_corr": mean_abs_pair_corr(lib.values, [x["id"] for x in found]),
        "novelty": float(np.mean([abs(x["metrics"].get("max_ref_corr", 0)) for x in found])) if found else float("nan"),
        "valid_hold": float(np.mean([x > 0 for x in ov])) if ov else float("nan"),
        "valid_t2": float(np.mean([x >= 2 for x in ot])) if ot else float("nan"),
        "valid_mean_ic": float(np.mean(ov)) if ov else float("nan"),
        "comp_found": vf.get("res_ic"), "comp_found_ir": vf.get("res_ir"),
        "comp_all_minus_seeds": (va.get("res_ic") - vs.get("res_ic")) if va.get("res_ic") is not None and vs.get("res_ic") is not None else None,
    }


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main():
    val = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else {}
    names = sys.argv[2:] or sorted(val)
    rows = [session_row(s, val) for s in names]
    cols = ["session", "n", "found", "admit_events", "weak", "redundant", "invalid", "dup", "tokens_k", "llm_min",
            "wall_min", "mechs", "pair_corr", "novelty", "valid_hold", "valid_t2", "valid_mean_ic", "comp_found",
            "comp_found_ir", "comp_all_minus_seeds"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r[c], 4 if c.startswith(("valid_mean", "comp")) else 2) if not isinstance(r[c], str)
                                else r[c] for c in cols) + " |")
    print("\nMean by arm:")
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    for arm, rs in by.items():
        agg = {c: np.nanmean([np.nan if r[c] is None else r[c] for r in rs]) for c in cols[1:]}
        print(f"  {arm:9s} " + " ".join(f"{c}={fmt(float(agg[c]), 4 if c.startswith(('valid_mean', 'comp')) else 2)}"
                                        for c in cols[1:]))


if __name__ == "__main__":
    main()
