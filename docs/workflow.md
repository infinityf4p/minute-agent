# Workflow 结构与用法

> 本文说明 v2 workflow 是怎么搭起来的、谁能改什么、怎么运行。实验结果见 [reports/v2_final_report.md](../reports/v2_final_report.md)。

## 1. 一句话

人设定开关和成本 → Agent（或随机搜索）在固定的动作空间里提出完整交易流程 → 固定的回测引擎在训练期两折上打分并反馈 → 评估器在验证期终选 → 测试期只打开一次做最终检验。

## 2. 分层

| 层 | 内容 | 谁能改 | 代码 |
|---|---|---|---|
| L0 数据与规则 | 分钟线、日线、复权、按时点推断的 ST、涨跌停价、交易日历、数据划分 | 人；锁定测试期由代码守卫 | `data/`、`split.py` |
| L1 信号 | 12 个特征算子 + Agent 自写的沙箱特征函数；只用决策时点及之前的 bar | Agent（在白名单内） | `features/` |
| L2 目标与模型 | 预测目标菜单、6 种模型及超参范围、滚动训练窗口 | Agent（在菜单内） | `pipeline/`、`models.py` |
| L3 策略模板与仓位 | 4 个模板及其参数 | Agent（在模板内），受两个开关约束 | `pipeline/spec.py` |
| L4 执行与成本 | 下一根 bar 成交、参与率上限、涨跌停不成交、T+1、T0 强制回补、费率滑点 | 不可改；费率由人在 `configs/costs/` 设置 | `backtest/` |
| L5 评估与终选 | 指标、试验账本、两级评估、验证期终选、DSR / Holm、测试期一次性打开 | 不可改 | `evaluator/`、`report/`、`scripts/v2_*.py` |

L4 和 L5 的源码有哈希锁（`evaluator/FROZEN.sha256`）。改动后运行器会拒绝执行，必须由人重新冻结。

## 3. 人设置的开关

在每个搜索会话启动时设置，Agent 只能看到、不能改。

| 开关 | 关闭时 | 打开时 |
|---|---|---|
| `allow_t0` | 只能用 T1、T3 | 还可以用 T2、T4（底仓 T0） |
| `position_sizing` | 满仓等权；T0 固定比例下单 | 可选权重方式（等权 / 分数加权 / 逆波动）、目标波动率、按趋势降仓；T0 按信号强弱调整下单量 |
| 成本（`configs/costs/*.yaml`） | 佣金、最低佣金、过户费、按日期切换的印花税、滑点、冲击系数、参与率上限 | |
| 账户规模、底仓股票数 | 默认 1 亿元、300 只 | |

## 4. 策略模板

| 模板 | 做什么 | 决策 | 优化目标 |
|---|---|---|---|
| T1 横截面多头 | 选分数最高的股票持有多日，可设缓冲带、最短持有期、换手上限、止损 | 每天一次，时点从 09:45–14:50 中选 | 对冲超额净 Sharpe |
| T3 尾盘→隔夜 | 决策后买入，次日开盘集合竞价卖出 | 每天一次 | 对冲超额净 Sharpe |
| T2 底仓 T0 | 固定底仓（每月按成交额选前 300 只、等权）上做 T：日内先卖后买 / 先买后卖并在 14:45 起强制回补；或隔夜版（收盘前卖、次日开盘买回） | 日内每 15 或 30 分钟一次；隔夜版每天 14:50 | 按底仓市值计的叠加收益（bp/天） |
| T4 选股 + T0 | T1 的持仓作为底仓，再叠加 T0 | 两者 | 合并后的净 Sharpe |

## 5. 回测引擎的固定规则

- 在第 t 根 bar 收盘后决策，在之后的 bar 成交。成交价可选：下一根 bar 的成交均价、未来 5 根或 10 根 bar 的均价。绝不在信号所在 bar 成交。
- 每笔成交不超过成交 bar 成交量的 10%（参与率上限可在成本配置里改）。
- 一字涨停不能买，一字跌停不能卖；成交量为零的 bar 不成交。
- 选股只能用决策时点已知的信息（例如当时是否已在涨停价），能否成交由之后的 bar 决定，不能事后换股。
- 当天买入的股票当天不能卖（T+1）。T0 卖出量不超过底仓，14:45、14:50、14:55 依次尝试回补，允许部分成交，还没回补的部分按次日开盘价结算。
- 规则按日期切换：印花税 2023-08-28 从 0.1% 降为 0.05%；主板 ST 涨跌幅 2026-07-06 从 5% 调为 10%。
- 每个结果都在三档成本下计算：基准成本、2 倍成本、不计成本。

## 6. 数据划分与评估流程

| 阶段 | 训练数据 | 评估数据 | 谁能看到 |
|---|---|---|---|
| 初筛 `screen` | 2020 年及之前 | 2021 | Agent |
| 两折 `folds` | 每折只用评估年之前的数据 | 2020、2021 | Agent |
| 验证 `valid` | 2023 年及之前（滚动） | 2022–2023 | 只有评估器 |
| 测试 `test` | 2026 年 10 月及之前（滚动） | 2024-01 到 2026-10 | 只打开一次 |

- 每个提案先初筛；初筛成绩在同模板里排前 30%（或该模板前 3 次）才跑两折。两次评估都计入预算和试验次数。
- 读数据不超过当前阶段的结束日，所以 2021 年末需要用到 2022 年数据的标签会被丢弃，不会偷看验证期。
- 每个会话结束后，评估器取每个模板两折成绩前 3 的配置跑验证期，再选验证期最好的作为该模板的终选。
- 全部会话的终选和固定基线在测试期一次性评估，做均值检验和 Holm 校正，并按 2026-07-06 前后分段报告。

## 7. 怎么运行

```bash
# 一个搜索会话（可用 --name 续跑）
conda run -n minute-agent python scripts/v2_search.py agent --budget 100 --seed 0 --t0 --sizing
conda run -n minute-agent python scripts/v2_search.py random --budget 100 --seed 0 --t0 --sizing

# 单个配置在某个阶段上跑一次（调试用）
conda run -n minute-agent python scripts/run_one.py my_config.json folds --t0 --sizing

# 验证期终选（只由评估器运行）
conda run -n minute-agent python scripts/v2_validate.py p4-agent-s0 p4-random-s0

# 测试期（只能一次；需要设置解锁环境变量）
conda run -n minute-agent python scripts/v2_final.py p4-agent-s0 p4-random-s0
```

长任务建议用 `scripts/launch.ps1` 独立启动，这样 Claude Code 重启也不会中断。

## 8. 一个配置长什么样

```json
{
  "template": "T2",
  "target": {"kind": "overnight"},
  "t0": {"mode": "overnight", "direction": "sell_first", "size_frac": 0.5, "entry_q": 0.9},
  "features": [{"op": "ret", "window_min": [5, 30, 60, 240]}, {"op": "vwap_dev", "at": "decision_time"}],
  "label_norm": "rank", "cross_section_norm": "rank",
  "model": {"type": "xgb", "params": {"max_depth": 6, "lr": 0.05, "n_estimators": 200}},
  "train": {"window_years": 3, "retrain_every": "quarter"},
  "execution": {"style": "next_vwap"},
  "universe_filter": {"min_price": 2, "min_adv_pct": 20}
}
```

完整的取值范围见 `pipeline/spec.py` 的 `describe()`，Agent 看到的就是它的输出。

## 9. 已知限制

- 数据不含已退市股票（幸存者偏差），ST 状态靠价格推断。
- 只有 1 分钟 K 线，没有盘口和逐笔，T0 的效果应预期弱于业界宣传。
- 更高频的维度（5 分钟或 1 分钟决策、日内多次调仓）暂未开放，计划在测试阶段之后再加。
- 北交所未纳入。
