"""Collect the numbers of the v2 experiment into reports/v2_results.md (tables only; the narrative is written by hand).

Reads the ledger (search phase), runs/v2_validation.json (validation) and runs/final/v2_test.json (test).

    conda run -n minute-agent python scripts/v2_report.py p4-agent-s0 p4-agent-s1 ... p4-random-s2
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minute_agent.experiment.ledger import Ledger  # noqa: E402
from minute_agent.paths import runs_root  # noqa: E402

TPLS = ["T1", "T2", "T3", "T4"]


def f(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def counted(r):
    return not (r["status"] == "failed" and (r["error"] or "").startswith("interrupted"))


def main():
    sessions = sys.argv[1:]
    led = Ledger()
    md = ["# v2 实验数据表", "", "由 `scripts/v2_report.py` 生成。", ""]

    # 1. search phase
    md += ["## 1. 搜索阶段（训练期：初筛 2021，两折 2020/2021）", ""]
    rows, per_arm = [], defaultdict(lambda: defaultdict(list))
    for s in sessions:
        rs = [r for r in led.query("SELECT * FROM runs WHERE session=?", (s,)) if counted(r)]
        meta = json.loads((runs_root() / "sessions" / s / "session.json").read_text())
        arm = meta["arm"]
        tpl = Counter()
        best = {}
        for r in rs:
            t = json.loads(r["config_json"])["cfg"].get("template")
            if r["eval_split"] == "screen":
                tpl[t] += 1
                if r["status"] == "done" and r["objective"] is not None:
                    per_arm[arm][t].append(r["objective"])
            if r["eval_split"] == "folds" and r["status"] == "done" and r["objective"] is not None:
                best[t] = max(best.get(t, -1e9), r["objective"])
        fails = sum(r["status"] == "failed" for r in rs)
        folds = sum(r["eval_split"] == "folds" for r in rs)
        secs = [r["t_total"] for r in rs if r["t_total"]]
        tok = ""
        u = runs_root() / "sessions" / s / "llm_calls.jsonl"
        if u.exists():
            n = sum(json.loads(line)["usage"]["total_tokens"] for line in u.open(encoding="utf-8"))
            tok = f"{n / 1e6:.2f}M"
        rows.append([s, len(rs), folds, fails, " / ".join(f"{t}:{tpl[t]}" for t in TPLS),
                     " / ".join(f"{t}:{f(best.get(t))}" for t in TPLS), f"{np.median(secs):.0f}s", tok or "–"])
    md += [table(["会话", "评估数", "其中两折", "失败", "初筛次数（按模板）", "两折最好目标值", "单次耗时中位数", "LLM tokens"], rows), ""]
    md += ["目标值：T1/T3/T4 为年化 Sharpe，T2 为 bp/天。", ""]

    rows = []
    for t in TPLS:
        for arm in ("agent", "random"):
            v = np.array(per_arm[arm][t])
            if len(v):
                rows.append([t, arm, len(v), f(np.median(v)), f(np.quantile(v, 0.9)), f(v.max()), f"{np.mean(v > 0):.0%}"])
    md += ["### 初筛目标值分布（三个种子合并）", "", table(["模板", "臂", "次数", "中位数", "90 分位", "最大", "为正比例"], rows), ""]

    # 2. validation
    val = json.loads((runs_root() / "v2_validation.json").read_text())
    md += ["## 2. 验证期终选（2022–2023，只有评估器可见）", ""]
    rows = []
    for s in sessions:
        for t in TPLS:
            x = val[s]["templates"].get(t)
            if not x:
                continue
            c = next(c for c in x["candidates"] if c["valid_run"] == x["finalist"])
            rows.append([s, t, x["n_trials"], f(c["folds_objective"]), f(c["valid_objective"]), f(c["valid_x2_objective"]),
                         f(c["valid_bp_per_day"]), f((c["dsr"] or {}).get("dsr"), 3, False)])
    md += [table(["会话", "模板", "该模板试验数 N", "两折目标", "验证目标", "验证目标（2 倍成本）", "验证 bp/天", "DSR"], rows), ""]

    # 3. test
    p = runs_root() / "final" / "v2_test.json"
    if p.exists():
        test = json.loads(p.read_text())
        holm = test.get("holm", {})
        md += ["## 3. 测试期（2024-01 至 2026-10，只打开一次）", ""]
        rows = []
        for e in test["entries"]:
            if e["status"] != "done":
                rows.append([e["label"], e.get("arm"), "失败", e.get("error", "")[:60], "", "", "", "", "", "", "", ""])
                continue
            m = e["metrics"]
            h = holm.get(e["label"], {})
            rows.append([e["label"], e["arm"], f(e.get("valid_objective")), f(m.get("objective")), f(m.get("x2_objective")),
                         f(m.get("gross_objective")), f(m.get("mean_bp_per_day")), f(e["t"]), f(e["p"], 4, False),
                         f(h.get("p_holm"), 4, False), f((e.get("dsr") or {}).get("dsr"), 3, False),
                         f"{f(e.get('bp_per_day_before_2026_07_06'))} / {f(e.get('bp_per_day_after_2026_07_06'))}"])
        md += [table(["流程", "臂", "验证目标", "测试目标", "测试目标（2 倍成本）", "测试目标（不计成本）", "bp/天", "t", "p",
                      "Holm p", "DSR", "bp/天 2026-07-06 前 / 后"], rows), ""]
        done = [e for e in test["entries"] if e["status"] == "done" and e.get("valid_objective") is not None]
        by_t = defaultdict(list)
        for e in done:
            by_t[e["label"].split(":")[-1]].append(e)
        rows = []
        for t, es in sorted(by_t.items()):
            for arm in ("agent", "random"):
                v = [e["metrics"]["objective"] for e in es if e["arm"] == arm]
                if v:
                    rows.append([t, arm, len(v), f(np.mean(v)), " ".join(f(x) for x in v)])
        md += ["### agent 与随机：测试期目标值", "", table(["模板", "臂", "终选数", "均值", "各种子"], rows), ""]
        if len(done) > 3:
            vo = np.array([e["valid_objective"] for e in done]).argsort().argsort()
            to = np.array([e["metrics"]["objective"] for e in done]).argsort().argsort()
            rho = float(np.corrcoef(vo, to)[0, 1])
            md += [f"验证目标与测试目标的 Spearman 相关（全部终选）：{rho:+.2f}", ""]

    out = REPO / "reports" / "v2_results.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("saved", out)


if __name__ == "__main__":
    main()
