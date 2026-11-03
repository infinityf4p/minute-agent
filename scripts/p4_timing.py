"""Where does the time go? Per-stage timings of P4 evaluations from the ledger."""
import json
import sqlite3
import statistics as st

c = sqlite3.connect("runs/ledger.sqlite")
rows = c.execute("SELECT session, eval_split, status, t_total, metrics_json, config_json FROM runs WHERE session LIKE 'p4-%'").fetchall()
agg = {}
for s, ph, status, tt, mj, cj in rows:
    if status != "done":
        continue
    t = json.loads(mj).get("timings", {})
    tpl = json.loads(cj)["cfg"]["template"]
    for k in ("features", "panel", "train", "backtest", "total"):
        agg.setdefault((ph, k), []).append(t.get(k, 0.0))
    agg.setdefault(("tpl", tpl), []).append(t.get("total", 0.0))
fails = sum(1 for r in rows if r[2] == "failed")
print(f"done {sum(1 for r in rows if r[2]=='done')}, failed {fails}")
for (a, b), v in sorted(agg.items()):
    print(f"{a:6s} {b:9s} n={len(v):3d} median {st.median(v):6.0f}s  mean {st.mean(v):6.0f}s  max {max(v):6.0f}s  sum {sum(v)/60:5.0f}min")
