# minute-agent

一个独立的 LLM Agent，在 A 股和美股的分钟数据上，自动搭建并测试机器学习因子流程。

- [CONTEXT.md](CONTEXT.md)：项目背景、调研结论、已经做出的决定。新会话请先读这份。
- [PLAN.md](PLAN.md)：完整计划，包括实验设定、架构、动作空间、评估方法、里程碑和数据清单。

行情数据、特征缓存和实验结果都不进 git，规则见 `.gitignore`。

## 快速开始（Windows）

```bash
conda env create -f environment.yml
conda run -n minute-agent pip install -e . --no-deps
conda run -n minute-agent python -m pytest -q
conda run -n minute-agent python -m minute_agent.cli sim --n-stocks 500        # 生成模拟数据到 data/sim_cn
conda run -n minute-agent python -m minute_agent.cli run configs/experiments/baseline_lgbm.yaml
conda run -n minute-agent python -m minute_agent.cli ledger                    # 查看实验账本
```

注意：一定要通过 `conda run -n minute-agent`（或先 `conda activate`）运行。直接调用环境里的 `python.exe` 会加载 base 环境的 DLL，numpy 的线性代数会崩溃（0xc06d007f）。

## 代码结构

| 路径 | 内容 |
|---|---|
| `src/minute_agent/split.py` | 训练/验证/锁定测试集划分，以及读取锁定期的守卫 |
| `src/minute_agent/data/` | 标准 schema、parquet 读写（按市场/年/月分区）、A 股模拟数据生成器 |
| `configs/markets/` | 市场规则：T+1、做空、涨跌停、成本、股票池、组合类型 |
| `src/minute_agent/features/` | 分钟到日度的特征算子、防泄漏检查、parquet 特征缓存 |
| `src/minute_agent/panel.py` | 按时点构建的股票池和预测标签 |
| `src/minute_agent/evaluator/` | 固定评估器，源码有哈希锁（`FROZEN.sha256`），Agent 不能修改 |
| `src/minute_agent/experiment/` | 实验配置校验、滚动重训运行器、SQLite 实验账本 |
| `scripts/benchmark.py` | 测速：冷启动特征计算 + 一轮训练和评估 |

## 锁定测试集

2024-01-01 到 2026-10-31 的数据在机制上不可读：所有读取函数在打开文件前检查日期，超过 2023-12-31 直接报错，锁定期的分区不会被打开，证券信息表里未来的退市日期也会被隐藏。唯一的开关需要同时满足两个条件：环境变量 `MINUTE_AGENT_UNLOCK_TEST` 等于 `split.UNLOCK_PHRASE`，并且调用时显式传 `allow_locked=True`。实验配置里没有任何字段能打开它。

## 防泄漏

- 生产路径里，特征算子只拿到决策时点（默认 14:50）及之前的 bar。
- 每个特征算子第一次进缓存前，都要通过截断测试：在第 T 天的"信息集"（T 天之前的全部数据、T 天 14:50 前的分钟线、T 天开盘前已知的日线字段）上重算，结果必须和用全量数据算的一致。
- `tests/test_leakage.py` 里有 4 种故意作弊的特征（偷看次日开盘、当天日线收盘价、15:00 收盘竞价、全样本标准化），全部会被拦下。
