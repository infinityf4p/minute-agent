"""Validation period (2022-2023) of each session's library. The orientation of every factor is fixed by its
mining-period sign; nothing here feeds back into any search.

Per session:
  members      each found factor's residual IC on the validation period (oriented), per year, long-short
  composites   equal-weight mean of oriented centered ranks of
                 seeds  the 10 fixed reference factors (identical across sessions)
                 found  the factors the search found
                 all    seeds + found  (increment over seeds = what the search added)

    conda run --no-capture-output -n minute-agent python scripts/factormine/fm_validate.py pilot-random-s0 ... --out runs/factormine/validation_pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from minute_agent.factormine import config, data, evaluate
from minute_agent.factormine.library import Library

KEYS = ["symbol", "date"]


def oriented(cfg, spec, periods, sign) -> pl.DataFrame:
    fr = evaluate.factor_frame(cfg, spec, periods).filter(pl.col("f").is_not_null())
    return fr.select(KEYS + [(sign * ((pl.col("f").rank() / pl.col("f").count()).over("date") - 0.5)).alias("o")])


def composite_metrics(cfg, cols: dict[str, pl.DataFrame], periods) -> dict:
    if not cols:
        return {}
    j = None
    for k, o in cols.items():
        o = o.rename({"o": k})
        j = o if j is None else j.join(o, on=KEYS, how="full", coalesce=True)
    names = list(cols)
    j = j.with_columns(pl.mean_horizontal(names).alias("f"),
                       pl.sum_horizontal([pl.col(c).is_not_null() for c in names]).alias("_n"))
    j = j.filter(pl.col("_n") >= max(1, len(names) // 2)).select(KEYS + ["f"])
    start, end = config.span(periods)
    base = data.base_panel(cfg, data.shift_months(start, cfg["warmup_months"]), end)
    inside = pl.lit(False)
    for a, b in periods:
        inside = inside | pl.col("date").is_between(a, b)
    fr = base.filter(inside & pl.col("mask")).join(j, on=KEYS, how="left").select(
        KEYS + ["f", "label", "label_on"] + data.STYLES)
    m, _ = evaluate.metrics(fr, periods, sign=1.0)
    return {k: m.get(k) for k in ["coverage", "ic", "icir", "res_ic", "res_ir", "res_t", "res_ic_folds",
                                  "res_ic_overnight", "rank_autocorr", "ls_gross_bp", "ls_net_bp"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_p = Path(a.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for s in a.sessions:
        sd = config.runs_dir() / "sessions" / s
        meta = json.loads((sd / "session.json").read_text())
        cfg = config.get(meta["cfg"])
        V, M = cfg["valid"], cfg["mine"]
        lib = Library(sd / "library")
        members, cols_v, cols_m = [], {}, {}
        for x in lib.members:
            sign = float(x["metrics"]["sign"])
            ov = oriented(cfg, x["spec"], V, sign)
            cols_v[x["id"]] = ov
            cols_m[x["id"]] = lib.values.select(KEYS + [pl.col(x["id"]).alias("o")]).drop_nulls()
            if x.get("fixed"):
                continue
            fr = evaluate.factor_frame(cfg, x["spec"], V)
            mv, _ = evaluate.metrics(fr, V, sign=sign)
            members.append({"id": x["id"], "mechanism": x["spec"].get("mechanism"), "mine": x["metrics"],
                            "valid": {k: mv.get(k) for k in ["coverage", "res_ic", "res_t", "res_ic_folds", "ic",
                                                              "rank_autocorr", "ls_gross_bp", "ls_net_bp"]}})
            print(s, x["id"], f"mine res_ic {x['metrics']['res_ic']:+.4f} -> valid {mv.get('res_ic', float('nan')):+.4f} "
                  f"(oriented sign {sign:+.0f}, valid t {mv.get('res_t', float('nan')):+.1f})", flush=True)
        seeds = [x["id"] for x in lib.members if x.get("fixed")]
        found = [x["id"] for x in lib.members if not x.get("fixed")]
        out = {"arm": meta["arm"], "seed": meta["seed"], "n_found": len(found), "members": members}
        for tag, ids in (("seeds", seeds), ("found", found), ("all", seeds + found)):
            out[f"valid_{tag}"] = composite_metrics(cfg, {k: cols_v[k] for k in ids}, V)
            out[f"mine_{tag}"] = composite_metrics(cfg, {k: cols_m[k] for k in ids}, M)
        print(s, "composite valid res_ic: seeds", f"{out['valid_seeds'].get('res_ic', float('nan')):+.4f}",
              "found", f"{(out['valid_found'] or {}).get('res_ic', float('nan')):+.4f}",
              "all", f"{out['valid_all'].get('res_ic', float('nan')):+.4f}", flush=True)
        res[s] = out
        out_p.write_text(json.dumps(res, indent=1, default=lambda v: None if isinstance(v, float) and not np.isfinite(v) else str(v)))
    print("saved", out_p)


if __name__ == "__main__":
    main()
