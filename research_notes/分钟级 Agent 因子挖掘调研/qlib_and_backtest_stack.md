# Qlib Research Workflow, High-Frequency / Nested Execution, QlibRL, and Other Minute-Level Backtesting Stacks

Research date: 2026-11-02. All facts were taken from the URLs cited inline. Parameter names are quoted exactly as they appear in source code or docs. Version notes:
- GitHub releases list **Qlib v0.9.8** (dated "08 Oct", year not shown on the page) as the latest release, with security hardening — [GitHub releases](https://github.com/microsoft/qlib/releases).
- The PyPI page fetched on the same day still shows **pyqlib 0.9.7 (15 Aug 2025)** as latest, supporting Python 3.8–3.12 — [PyPI pyqlib](https://pypi.org/project/pyqlib/).
- These two sources conflict. v0.9.8 may be on GitHub without a matching PyPI release, or the PyPI page may have been cached. Check with `pip index versions pyqlib` before pinning.

---

## Q1. How does a Qlib qrun YAML pipeline fit together (data → handler/processors → dataset → model → records), and which parts are usually swapped?

### Takeaway
A Qlib `qrun` YAML has four main blocks:
- `qlib_init` sets the data provider and region.
- `task.dataset` is a `DatasetH` that wraps a DataHandler (`Alpha158`/`Alpha360`) and named time `segments`.
- `task.model` is a model-zoo class.
- `task.record` is a chain of `SignalRecord` → `SigAnaRecord` → `PortAnaRecord`. `PortAnaRecord` embeds the strategy, executor and exchange config.

In typical research, people change the handler (features/label), the processors, the model class and kwargs, and the segment dates. The record and backtest blocks usually stay fixed.

### Cited Findings
- The official LightGBM/Alpha158 benchmark YAML has these top-level keys: `qlib_init` (`provider_uri: "~/.qlib/qlib_data/cn_data"`, `region: cn`), `market: csi300`, `benchmark: SH000300`, `data_handler_config`, `port_analysis_config` and `task` — [workflow_config_lightgbm_Alpha158.yaml](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- `data_handler_config` has `start_time: 2008-01-01`, `end_time: 2020-08-01`, `fit_start_time: 2008-01-01`, `fit_end_time: 2014-12-31` and `instruments: *market`. Processors that learn statistics (for example z-score) are fitted on the `fit_*` window — [same YAML](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- The dataset is `class: DatasetH`, `module_path: qlib.data.dataset`, with `handler: {class: Alpha158, module_path: qlib.contrib.data.handler}` and `segments: train [2008-01-01, 2014-12-31]`, `valid [2015-01-01, 2016-12-31]`, `test [2017-01-01, 2020-08-01]` — [same YAML](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- The model is `class: LGBModel`, `module_path: qlib.contrib.model.gbdt`, with kwargs `loss: mse`, `colsample_bytree: 0.8879`, `learning_rate: 0.2`, `subsample: 0.8789`, `lambda_l1: 205.6999`, `lambda_l2: 580.9768`, `max_depth: 8`, `num_leaves: 210`, `num_threads: 20` — [same YAML](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- The records are three classes from `qlib.workflow.record_temp`, connected by placeholders — [same YAML](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
  - `SignalRecord` (kwargs `model: <MODEL>`, `dataset: <DATASET>`)
  - `SigAnaRecord` (kwargs `ana_long_short: False`, `ann_scaler: 252`)
  - `PortAnaRecord` (kwargs `config: *port_analysis_config`)
  - The strategy receives the prediction through `signal: <PRED>`.
- `port_analysis_config` in that YAML — [same YAML](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml):
  - strategy: `class: TopkDropoutStrategy`, `module_path: qlib.contrib.strategy`, `topk: 50`, `n_drop: 5`
  - backtest: `start_time: 2017-01-01`, `end_time: 2020-08-01`, `account: 100000000`, `benchmark: *benchmark`
  - `exchange_kwargs`: `limit_threshold: 0.095`, `deal_price: close`, `open_cost: 0.0005`, `close_cost: 0.0015`, `min_cost: 5`
- Handler constructor arguments shared by `Alpha158` and `Alpha360` — [qlib/contrib/data/handler.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py):
  - `instruments="csi500"`, `start_time=None`, `end_time=None`, `freq="day"`, `fit_start_time=None`, `fit_end_time=None`, `filter_pipe=None`, `inst_processors=None`
  - also `infer_processors`, `learn_processors` and `**kwargs`
  - `Alpha158` alone also takes `process_type=DataHandlerLP.PTYPE_A`.
- Default processors — [handler.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py):
  - **Alpha158**: `infer_processors=[]`.
  - **Alpha360**: `_DEFAULT_INFER_PROCESSORS`, which is `ProcessInf`, then `ZScoreNorm`, then `Fillna`.
  - **Both**: `_DEFAULT_LEARN_PROCESSORS`, which is `{"class": "DropnaLabel"}` then `{"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}`.
  - In other words, the label is cross-sectionally z-scored only for learning, and inference data is left alone.
- The default label for both handlers is `"Ref($close, -2)/Ref($close, -1) - 1"` (named `LABEL0`). This is the return from T+1 close to T+2 close, which matches A-share T+1 trading. `Alpha158vwap` and `Alpha360vwap` use `Ref($vwap, -2)/Ref($vwap, -1) - 1` instead — [handler.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py)
- Release v0.9.7 added Parquet data support, a `BaseDataHandler` refactor, a geometric accumulation mode for `risk_analysis` and a draft Data Health Checker. v0.9.6 added a nested data loader — [GitHub releases](https://github.com/microsoft/qlib/releases)

### Inferences
- The "infer vs learn" processor split is the main lever for preventing leakage. Normalization that uses future cross-sections (`CSZScoreNorm` on features) is safe within one timestamp. Time-series `ZScoreNorm` must be fitted only on `fit_start_time`–`fit_end_time`. A fixed evaluator should pin those windows.
- In an agent factor-mining loop, the natural "agent-editable surface" is the handler's feature expressions, the processors and the model kwargs. `segments`, `port_analysis_config`, the records and `exchange_kwargs` should be frozen by the harness.
- The default daily label skips the next day (`Ref(-2)/Ref(-1)`). A minute-level workflow needs an equivalent "decision-time to executable-time" offset in its label.

### Gaps
- I did not fetch the qrun CLI docs. `qrun <yaml>` is the documented entry point in the Qlib README, but I did not re-verify the flags.
- I did not find a built-in high-frequency DataHandler class name (for example `HighFreqHandler`). The `examples/highfreq` README does not name it (see Q3). It is defined in the example's own files, which I did not fetch.

---

## Q2. What are Qlib's strategy-layer interfaces (TopkDropout, WeightStrategyBase, EnhancedIndexing) and Exchange knobs (deal_price, limits, costs, trade_unit, T+1)?

### Takeaway
Qlib's strategies turn a score signal into orders. The `Exchange` then simulates fills with a small, fixed set of rules:
- a fixed or (buy, sell) deal-price field
- a limit rule given as a float or two expressions
- optional volume caps (`"cum"` or `"current"`)
- asymmetric proportional costs with a minimum plus a quadratic impact term
- rounding to `trade_unit` lots

There is no explicit T+1 position lock in the Exchange. The nearest knob is the strategy's `hold_thresh`.

### Cited Findings
**TopkDropoutStrategy** ([signal_strategy.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/strategy/signal_strategy.py))
- Parameters:
  - `topk`: "the number of stocks in the portfolio"
  - `n_drop`: "number of stocks to be replaced in each trading date"
  - `method_sell`: "random/bottom"
  - `method_buy`: "random/top"
  - `hold_thresh`: "minimum holding days"
  - `only_tradable`
  - `forbid_all_trade_at_limit`
  - `risk_degree` is inherited from `BaseSignalStrategy` and means "position percentage of total value".
- Algorithm:
  1. Take the previous step's signal (first column only).
  2. Choose buy candidates from unheld names; the number is `n_drop + topk - len(held)`.
  3. Rank held names together with candidates, and sell the bottom `n_drop` held names.
  4. Simulate the sells first and add their proceeds to cash.
  5. Give each buy `cash * risk_degree / len(buy)`, rounded to the trade unit.
- `hold_thresh`: before selling, the strategy checks `get_stock_count(code, bar=time_per_step) >= hold_thresh`. The default of 1 rarely blocks a sell.
- `forbid_all_trade_at_limit=True` passes `direction=None` to `is_stock_tradable`, so nothing is traded at limit-up or limit-down. With `False`, the docstring says the "strategy will sell at limit up and buy ad[sic] limit down".
- `only_tradable=True` filters untradable stocks out at candidate selection. Tradability is still checked again for every order either way.

**WeightStrategyBase** ([same file](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/strategy/signal_strategy.py))
- Subclasses implement `generate_target_weight_position`.
- `order_generator_cls_or_obj` defaults to `OrderGenWOInteract`. It converts target weights times `risk_degree` into orders.

**EnhancedIndexingStrategy** ([same file](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/strategy/signal_strategy.py))
- Parameters:
  - `riskmodel_root`: risk data is read from a `YYYYMMDD` subfolder.
  - `market="csi500"`: the benchmark weight field is `${market}_weight`.
  - `turn_limit`: stored on the instance but not used in the code read.
  - `name_mapping`: alternative file names for factor exposure, factor covariance, specific risk and blacklist.
  - `optimizer_kwargs`: passed to `EnhancedIndexingOptimizer`.
  - `verbose`
- The class docstring documents a stale `riskmodel_path`, which does not match the actual parameter.

**Exchange.__init__** ([qlib/backtest/exchange.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/backtest/exchange.py))
- Parameters and defaults:
  - `freq="day"`, `start_time`, `end_time`, `codes="all"`
  - `deal_price=None` (falls back to `C.deal_price`). A single field applies to both sides, and a `(buy, sell)` tuple sets them separately. `$` is auto-prefixed.
  - `subscribe_fields=[]`
  - `limit_threshold=None` (falls back to `C.limit_threshold`)
  - `volume_threshold=None`
  - `open_cost=0.0015`, `close_cost=0.0025`, `min_cost=5.0`
  - `impact_cost=0.0` (the docstring says "A recommended value is 0.1")
  - `extra_quote=None` (must contain `$close`)
  - `quote_cls=NumpyQuote`
  - `trade_unit` is popped from kwargs. The docstring says "100 for China A market", and `None` disables it. Any other unknown kwarg raises `ValueError`.
- `limit_threshold` modes:
  - `None`: only suspended stocks (NaN `$close`) are blocked.
  - float: a buy is blocked if `$change >= threshold`, and a sell is blocked if `$change <= -threshold`.
  - tuple of two expressions: the first gives the buy-limited flag and the second the sell-limited flag; each is OR'ed with suspension.
- `volume_threshold`: entries are `(mode, expression)` with mode `"cum"` or `"current"`, keyed by `"buy"`, `"sell"` or `"all"`.
  - `"cum"` subtracts volume already dealt for that stock.
  - When several caps apply, the minimum wins, and the order is clipped to it.
- Cost model:
  - `adj_cost_ratio = impact_cost * (trade_val / total_trade_val) ** 2`, where `total_trade_val = volume * trade_price`.
  - `cost_ratio` is `open_cost` or `close_cost` plus `adj_cost_ratio`.
  - `trade_cost = max(trade_val * cost_ratio, min_cost)`.
- Lot rounding: `(amount * factor + 0.1) // trade_unit * trade_unit / factor`. Rounding is disabled if any `$factor` is NaN while `$close` exists.
- If the deal price is NaN or ≤1e-8, the Exchange logs a warning and falls back to `$close`.

**T+1**
- A Qlib issue asks about entering at T+1 open and exiting at the same day's close. It was marked stale, and I saw no maintainer answer — [qlib issue #432](https://github.com/microsoft/qlib/issues/432)
- A separate PR adds short selling through `ShortableExecutor` and `ShortableAccount`, not T+1 settlement — [qlib PR #1986](https://github.com/microsoft/qlib/pull/1986/files)

### Inferences
- Qlib's float `limit_threshold` checks `$change` (close-to-previous-close), not the intraday price against the limit price.
  - For minute bars, a float threshold is wrong: a stock can touch the limit intraday and reopen.
  - The tuple-of-expressions mode lets you supply accurate per-bar buy-block and sell-block flags. Examples: `$close >= $limit_up` for buys; also STAR/ChiNext ±20%, ST ±5%, and BSE ±30% boards.
  - A fixed backtester should precompute these flags in the data layer, not in agent code.
- Default `open_cost`/`close_cost` in `Exchange` (0.15%/0.25%) differ from the benchmark YAML (0.05%/0.15%). Every run should state its costs explicitly.
- T+1 is not enforced by the Exchange code I read. In a daily TopkDropout it is implicitly satisfied, because shares bought today are only sold on a later step. In a nested intraday setup, a custom inner strategy could sell same-day buys unless the account or executor enforces a "sellable quantity". I did not verify this; it is a strong reason to build T+1 into a custom fixed engine.

### Gaps
- I did not verify whether Qlib's `Account`/`Position` tracks a "sellable today" quantity. No source I read mentions one.
- I did not fetch `OrderGenWOInteract` or `OrderGenWInteract` internals.

---

## Q3. How does Qlib's nested decision execution combine a daily portfolio strategy with intraday execution, how is 1-min data prepared, and which metrics come out?

### Takeaway
Nesting works through executors:
- A `NestedExecutor` with `time_per_step: "day"` runs the portfolio strategy, for example TopkDropout.
- Its `inner_strategy` (for example `SBBStrategyEMA`) splits each daily order across an inner `NestedExecutor` at `"30min"`.
- That executor's `TWAPStrategy` splits further into a `SimulatorExecutor` at `"5min"`. The exchange itself runs on `freq: "1min"` data.

Every level produces portfolio metrics plus execution indicators: FFR (fill rate), PA (price advantage vs a TWAP/VWAP base price) and POS (share of orders with positive PA).

### Cited Findings
- Design: each level has a Trading Agent (Information Extractor + Forecast Model + Decision Generator) and an Execution Env. The env "can be nested with finer-grained trading algorithm and execution environment inside", for example daily orders split into intraday decisions. Level frequencies can be customized, and strategies at different levels can be optimized jointly with QlibRL — [Qlib docs: Design of Nested Decision Execution Framework](https://qlib.readthedocs.io/en/latest/component/highfreq.html)
- Related examples:
  - `examples/nested_decision_execution/workflow.py`
  - `examples/highfreq` (prediction from high-frequency data)
  - `examples/orderbook_data/` (feature extraction without a fixed frequency)
  - the `high-freq-execution` branch of the repository, which holds the paper code
  - Source: [Qlib docs highfreq](https://qlib.readthedocs.io/en/latest/component/highfreq.html)
- Data preparation in the nested example — [nested workflow.py](https://raw.githubusercontent.com/microsoft/qlib/main/examples/nested_decision_execution/workflow.py):
  - `GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", region=REG_CN, version="v2", exists_skip=True)` for daily data
  - `GetData().qlib_data(target_dir=<HIGH_FREQ provider_uri>, interval="1min", region=REG_CN, version="v2", exists_skip=True)` for 1-min data
  - `qlib.init(provider_uri={"1min": ..., "day": "~/.qlib/qlib_data/cn_data"}, dataset_cache=None, expression_cache=None)`
- Executor tree in that example — [nested workflow.py](https://raw.githubusercontent.com/microsoft/qlib/main/examples/nested_decision_execution/workflow.py):
  - Outer: `NestedExecutor` with `"time_per_step": "day"`, `"track_data": True`, `"generate_portfolio_metrics": True`, `indicator_config: {"show_indicator": True}`.
  - Its `inner_strategy` is `SBBStrategyEMA` (`qlib.contrib.strategy.rule_strategy`, kwargs `instruments=csi300`, `"freq": "1min"`).
  - Middle: `NestedExecutor` with `"time_per_step": "30min"` and `inner_strategy` `TWAPStrategy`.
  - Innermost: `SimulatorExecutor` with `"time_per_step": "5min"` and `"verbose": True`.
- Backtest settings in that example — [nested workflow.py](https://raw.githubusercontent.com/microsoft/qlib/main/examples/nested_decision_execution/workflow.py):
  - `start_time 2020-09-20`, `end_time 2021-05-20`, `account 100000000`, benchmark `SH000300`
  - `exchange_kwargs`: `"freq": "1min"`, `"limit_threshold": 0.095`, `"deal_price": "close"`, `"open_cost": 0.0005`, `"close_cost": 0.0015`, `"min_cost": 5`
  - Top-level strategy: `TopkDropoutStrategy` with `topk 50`, `n_drop 5`.
  - Analysis: `PortAnaRecord(recorder, port_analysis_config, indicator_analysis_method="value_weighted")`.
  - Helpers:
    - `collect_data()` streams the trade decision at every level.
    - `check_diff_freq()` asserts that daily-resampled `total_cost` from the 30-min report equals the 1-day report.
    - `backtest_only_daily()` replaces the tree with one `SimulatorExecutor` at `"day"` for comparison.
- Execution indicators (`Indicator` in [qlib/backtest/report.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/backtest/report.py)):
  - **ffr** (full fill rate) = `deal_amount / amount`.
  - **pa** (price advantage) = `sign * (trade_price / base_price - 1)`, set to 0 at the innermost layer.
  - **pos** = `(pa > 0).sum() / pa.count()`.
  - Also `deal_amount`, `value` and `count`.
  - Aggregation methods: `mean`, `amount_weighted` (weights |deal_amount|) and `value_weighted` (weights |trade_value|).
  - The base price comes from `deal_price`. `twap` gives every step equal weight; `vwap` weights by exchange volume.
- Portfolio metrics (`PortfolioMetrics` in [report.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/backtest/report.py)):
  - **return**, which is "without transaction fee"
  - **cost** ("transaction fee and slippage"), turnover, account, value, cash
  - **bench**: a string is a single instrument; a list or dict is the equal-weighted mean of `$close/Ref($close,1)-1`. It is compounded over each window.
- The `examples/highfreq` README describes the dataset as "an example for RL high frequency trading". Commands are `python workflow.py get_data` and `python workflow.py dump_and_load_dataset`. `DatasetH` is serializable to pickle. The README reports a LightGBM/Alpha158 high-frequency trend benchmark: IC 0.0349, ICIR 0.3805, Rank IC 0.0435, Rank ICIR 0.4724 — [examples/highfreq README](https://raw.githubusercontent.com/microsoft/qlib/main/examples/highfreq/README.md)

### Inferences
- With the nested framework, a daily alpha can be checked for whether it survives realistic intraday execution: compare `backtest_only_daily` to the nested run. That gap is a direct "implementation shortfall" metric.
- For a minute-level alpha, where decisions themselves are made intraday, the outer strategy would run at e.g. `"30min"`/`"5min"` with a `SimulatorExecutor` on `freq: "1min"` data. The `limit_threshold` caveat from Q2 matters even more here.
- `generate_portfolio_metrics` and `indicator_config` at each level give a fixed, engine-side metric set (FFR/PA/POS plus returns). These are suitable as non-tamperable evaluation outputs.

### Gaps
- I did not fetch the nested-decision-execution paper's arXiv page. The Qlib docs point only to the `high-freq-execution` branch. The paper title and arXiv ID are unverified here.
- The `examples/highfreq` handler classes and their exact label/feature definitions were not shown in the README.
- `SBBStrategyEMA` and `TWAPStrategy` kwargs beyond those listed above were not verified.
- `PortAnaRecord` options (`risk_analysis_freq`, `indicator_analysis_freq`) were not shown in the example.

---

## Q4. What does QlibRL offer for order execution (PPO, OPDS), and how is it run?

### Takeaway
QlibRL builds on Tianshou and Gym. It breaks an RL problem into these parts:
- a Simulator
- State and Action Interpreters
- a Reward
- a Policy (a Tianshou policy)
- a `TrainingVessel`
- a `Trainer` with `trainer.fit()`

It ships single-asset order execution (SAOE) simulators and reproduces PPO (IJCAI 2020) and OPDS (AAAI 2021) on 5-min HS300 data. The reported metric is PA vs a TWAP baseline.

### Cited Findings
- Scenarios: order execution (single-asset; multi-asset "with cross-asset dependencies and cash constraints") and portfolio construction — [QlibRL overview](https://qlib.readthedocs.io/en/latest/component/rl/overall.html)
- Components — [QlibRL framework](https://qlib.readthedocs.io/en/latest/component/rl/framework.html):
  - `EnvWrapper` (a `gym.Env`)
  - Simulators: `SingleAssetOrderExecution`, which is built on the Qlib backtest, realistic and slow; and `SimpleSingleAssetOrderExecution`, which skips trading limits and rounding and is fast
  - State interpreter, action interpreter, reward function
  - Policy, using Tianshou policies directly
  - `TrainingVessel`, which builds EnvWrappers and supports parallel training
  - Trainer, with an sklearn-style `trainer.fit()`
- Example data and commands — [examples/rl_order_execution README](https://raw.githubusercontent.com/microsoft/qlib/main/examples/rl_order_execution/README.md):
  - `python -m qlib.cli.data qlib_data --target_dir ./data/bin --region hs300 --interval 5min`
  - then `scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml`, `scripts/gen_training_orders.py`, `scripts/merge_orders.py`
  - Train: `python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest`
  - Backtest: `python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml`
  - Training tests use `SingleAssetOrderExecutionSimple`, which has no amount limit. Backtests use `SingleAssetOrderExecution`, which enforces trade-unit multiples, so results differ.
- Reported PA (mean ± std over the 10 best validation checkpoints) — [rl_order_execution README](https://raw.githubusercontent.com/microsoft/qlib/main/examples/rl_order_execution/README.md):
  - OPDS (PPO policy): 0.4785 ± 0.7815
  - OPDS (DQN policy): −0.0114 ± 0.5780
  - PPO: −1.0935 ± 0.0922
  - TWAP: ≈ 0.0
  - The README notes that its TWAP misses the last 5 minutes. The benchmark section is marked TBD.
- The OPD paper is "Universal Trading for Order Execution with Oracle Policy Distillation" by Fang, Ren, Liu, Zhou, Zhang, Bian, Yu and Liu (AAAI 2021, arXiv 2103.10860). An oracle teacher with perfect information distills a policy toward near-optimal execution; code is at seqml.github.io/opd — [arXiv 2103.10860](https://arxiv.org/abs/2103.10860)
- Release v0.9.4 added exploration noise for the RL training collector and a multi-pass portfolio analysis record — [GitHub releases](https://github.com/microsoft/qlib/releases)

### Inferences
- QlibRL is for execution of a given parent order. It does not generate minute-level alpha. For an A-share minute factor-mining agent, the more relevant ideas are its simulator/interpreter split and the realistic-vs-simple simulator gap, which is an example of the bias caused by simplified fills.
- The PA spread between policies (±0.78 std) shows that execution metrics on 5-min data are noisy. A fixed evaluator should report confidence intervals.

### Gaps
- I did not find a Qlib example of an intraday alpha trading strategy, as opposed to an execution strategy, in the sources fetched.
- Multi-asset order execution code paths and their names were not verified.

---

## Q5. How do other open-source stacks model intraday fills, partial fills, slippage, T+1/T+0 and price limits?

### Takeaway
Only RQAlpha natively models A-share rules: price-limit blocking, a volume-participation cap and minute matching modes. vn.py's CTA engine fills full volume at the open or the limit price, with no limits or T+1. Nautilus and hftbacktest offer the most rigorous microstructure (OHLC path heuristics, probabilistic and tiered fills, queue and latency models), but they target futures and crypto. Backtrader, zipline and vectorbt offer generic slippage and volume caps. For A-share minute work, T+1, board-specific limits and lot sizes have to be added in every framework except RQAlpha.

### Cited Findings
**RQAlpha (Ricequant)** — `sys_simulation` mod ([README](https://raw.githubusercontent.com/ricequant/rqalpha/master/rqalpha/mod/rqalpha_mod_sys_simulation/README.rst))
- `matching_type` (default `"current_bar"`):
  - Daily data: `"current_bar"` or `"vwap"`.
  - Minute data adds `"next_bar"` (next bar's open).
  - Tick data: `"last"`, `"best_own"`, `"best_counterparty"`, `"counterparty_offer"`.
- `price_limit=True`: blocks matching for securities at limit-up or limit-down.
- `volume_limit=True` with `volume_percent=0.25`: caps each bar's fill at 25% of market volume.
- `liquidity_limit=False`: tick backtests only.
- `inactive_limit=True`: no fills on zero-volume bars, in day and minute backtests.
- `slippage_model="PriceRatioSlippage"` or `"TickSizeSlippage"` (or a custom `BaseSlippage`), with `slippage=0`.
- `signal=False`: signal mode skips matching entirely.
- `management_fee=[]`.

**RQAlpha run options** ([run_algorithm docs](https://rqalpha.readthedocs.io/zh-cn/latest/intro/run_algorithm.html))
- `--frequency` accepts `1d` or `1m`. "Minute data isn't provided in the open-source version."
- `-me/--match-engine` accepts `current_bar` (current close) or `next_bar` (next open).
- `-cm` sets `commission_multiplier` (default 1). `--short-stock/--no-short-stock` controls short selling.
- **Parameter priority is: strategy code > CLI/run_* args > user config > defaults.** Strategy code can therefore override engine settings.

**VeighNa / vn.py `vnpy_ctastrategy` BacktestingEngine** ([backtesting.py](https://raw.githubusercontent.com/vnpy/vnpy_ctastrategy/master/vnpy_ctastrategy/backtesting.py))
- BAR-mode limit orders:
  - `long_cross_price = bar.low_price`, `short_cross_price = bar.high_price`
  - fill price is `min(order.price, bar.open_price)` for buys and `max(order.price, bar.open_price)` for sells
- Stop orders trigger on high or low and fill at `max(stop.price, open)` for buys and `min(stop.price, open)` for sells.
- Fills are always complete: `order.traded = order.volume`, `Status.ALLTRADED`.
- Costs live only in PnL: `slippage` is a flat per-unit amount (`trade.volume * size * slippage`), and `rate` multiplies turnover. Slippage does not change fill prices.
- `set_parameters` arguments: `vt_symbol`, `interval`, `start`, `end`, `rate`, `slippage`, `size`, `pricetick`, `capital`, `mode`, `risk_free`, `annual_days=240`, `half_life=120`.
- There is no T+1 restriction, no limit-up/limit-down check and no volume constraint.
- The CTA backtester GUI supports intervals 1m, 1h, d, w and tick, and computes daily mark-to-market PnL net of commission and slippage — [vn.py CTA backtester docs](https://www.vnpy.com/docs/cn/community/app/cta_backtester.html)

**NautilusTrader bar execution** ([bar-execution docs](https://nautilustrader.io/docs/latest/concepts/backtesting/bar-execution))
- Requires `bar_execution=True`, `BookType.L1_MBP` and time bars.
- Each bar becomes four synthetic updates, with volume split evenly and any remainder on the close.
- Default order is Open→High→Low→Close. `bar_adaptive_high_low_ordering=True` visits the extreme nearest the open first; the docs call this a heuristic.
- `ts_init` must be the bar close.
- Orders sent from `on_bar` settle at bar N's close when there is no latency model. "Using the current bar's open from `on_bar` introduces look-ahead", and there is no native next-bar-open fill mode.

**NautilusTrader FillModel** ([fill-models docs](https://nautilustrader.io/docs/nightly/concepts/backtesting/fill-models/))
- `prob_fill_on_limit=1.0`, `prob_slippage=0.0` (L1 only, one tick against the order), `random_seed`.
- Built-in variants: `BestPriceFillModel`, `OneTickSlippageFillModel`, `TwoTierFillModel`, `ThreeTierFillModel`, `LimitOrderPartialFillModel`, `VolumeSensitiveFillModel`, `CompetitionAwareFillModel` (`liquidity_factor=0.3`) and others.
- `liquidity_consumption=False` by default, which lets one displayed size fill more than one simulated order.

**hftbacktest** ([docs](https://hftbacktest.readthedocs.io/en/latest/))
- Tick-by-tick simulation with "Full order book reconstruction based on Level-2 Market-By-Price and Level-3 Market-By-Order feeds".
- Models feed and order latency, plus queue position.
- Python strategies run under Numba; the Rust live bot supports Binance Futures and Bybit.
- `BacktestAsset` setters include `power_prob_queue_model(2)`, `log_prob_queue_model2()`, `power_prob_queue_model3(3)` and `.no_partial_fill_exchange()` — [queue models tutorial](https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html)

**Backtrader** ([slippage docs](https://www.backtrader.com/docu/slippage/slippage/))
- Broker parameters: `slip_perc=0.0`, `slip_fixed=0.0`, `slip_open=False`, `slip_match=True` (cap at high/low), `slip_limit=True`, `slip_out=False`.
- Close orders get no slippage. Limit orders never fill worse than their limit.
- The navigation also lists "Cheat-On-Open" and "Volume Filling - Fillers" pages, which I did not fetch.

**zipline-reloaded** ([API reference](https://zipline.ml4trading.io/api-reference.html))
- `VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)` uses quadratic impact. Partial fills stay open and "will be passed again … in the next minute".
- `FixedSlippage(spread=0.0)`.
- Commission models: `PerShare(cost=0.001, min_trade_cost=0.0)`, `PerTrade`, `PerDollar(cost=0.0015)`.

**vectorbt** ([Portfolio API](https://vectorbt.dev/api/portfolio/base/))
- `from_signals`/`from_orders` take `price`, `fees`, `slippage` and `size` (default `np.inf`). Defaults set to `None` come from global settings.
- Examples pass `price=ohlcv['Open']` separately from `close`.
- The vectorbt.pro docs say orders exceeding cash are partially filled unless `allow_partial=False`. Note this is the commercial version — [vectorbt.pro orders & execution](https://vectorbt.pro/zh/features/backtesting/orders-and-execution/)

**Other**
- The PyPI package GeneralBacktest explicitly models A-share T+1. It warns that "sell-first" ordering can use unsettled sale proceeds, and it offers T+0 mode separately — [GeneralBacktest on PyPI](https://pypi.org/project/GeneralBacktest/)
- JoinQuant (聚宽) and Juejin (掘金): I found no official documentation describing minute-matching volume ratios or price-limit rules. A community post recommends `set_option("avoid_future_data", True)` and warns about survivorship bias from using the current ST list — [juejin.cn post](https://juejin.cn/post/7633251780277370923). This is not an official source.

### Inferences
- For A-share minute work, the most transferable default conventions are:
  - RQAlpha's `next_bar` matching (decide on bar t, fill at the open of t+1)
  - `volume_percent` around 0.1–0.25
  - `inactive_limit` and `price_limit`
  - Nautilus's explicit look-ahead rule: never fill at the current bar's open from a bar-close callback
- vn.py-style complete fills at the open or limit price, without volume caps or limits, are optimistic for illiquid A-shares and for limit-hitting names.
- Applied to A-shares, a minimal faithful fill model needs:
  - T+1 sellable quantity
  - board-specific limit flags per bar
  - 100-share lots (for STAR Market, a 200-share minimum with 1-share increments, which is a domain fact I did not verify here)
  - stamp duty on sells only
  - a minimum commission
  - a participation cap
  - next-bar fills
  - call-auction handling at 09:25 and 14:57–15:00

### Gaps
- I found no official JoinQuant or Juejin (MyQuant) documentation on minute matching. Their conventions remain unverified.
- I did not verify vn.py's portfolio-strategy backtester (`vnpy_portfoliostrategy`) or any A-share-specific vn.py module.
- vectorbt (open-source) slippage formula and `allow_partial` semantics were not confirmed from primary docs.
- I did not confirm the hftbacktest class names `RiskAdverseQueueModel` and `PartialFillExchange`, or the current version numbers of each framework.
- Backtrader's cheat-on-open and volume fillers are known to exist from the docs navigation, but I did not fetch their details.

---

## Q6. What are the best practices for making the evaluator independent of strategy code so an agent cannot tamper with it?

### Takeaway
The sources point to a "fixed engine + declarative spec + restricted evaluation" design. Qlib's own v0.9.8 hardening goes this way: it restricts executable artifact loading, requires explicit trust for config-driven imports and parses expressions with a restricted parser. Several frameworks show how strategy code can silently change engine settings (RQAlpha) or create look-ahead (Nautilus, Qlib deal price). These are the vectors an agent-proof harness must close.

### Cited Findings
- Qlib v0.9.8 security hardening — [GitHub releases](https://github.com/microsoft/qlib/releases):
  - "Restrict loading of executable recorder artifacts"
  - explicit trust required for "configuration-driven local `.py` imports"
  - feature-expression evaluation replaced with "a restricted parser"
- Qlib's `Exchange` rejects unknown kwargs with `ValueError`. Its fill rules (limit, volume, cost, lot) live entirely in the engine, separate from strategies — [exchange.py](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/backtest/exchange.py)
- In RQAlpha, strategy code takes precedence over CLI args, config files and defaults. A strategy can therefore override matching and slippage settings unless the harness blocks it — [RQAlpha run_algorithm](https://rqalpha.readthedocs.io/zh-cn/latest/intro/run_algorithm.html)
- RQAlpha `signal=True` skips matching entirely ("所有通过风控的订单将不进行撮合，直接产生交易"). One flag turns realistic fills into free fills — [RQAlpha sys_simulation README](https://raw.githubusercontent.com/ricequant/rqalpha/master/rqalpha/mod/rqalpha_mod_sys_simulation/README.rst)
- Nautilus warns that filling at the current bar's open from `on_bar` "introduces look-ahead". It requires `ts_init` at bar close so bars are not visible before they form — [Nautilus bar execution](https://nautilustrader.io/docs/latest/concepts/backtesting/bar-execution)
- Nautilus advises setting `random_seed` for reproducible probabilistic fills — [Nautilus fill models](https://nautilustrader.io/docs/nightly/concepts/backtesting/fill-models/)
- Qlib's nested example includes consistency checks: `check_diff_freq` asserts that 30-min and daily `total_cost` agree, and `backtest_only_daily` provides a reference run — [nested workflow.py](https://raw.githubusercontent.com/microsoft/qlib/main/examples/nested_decision_execution/workflow.py)
- QlibRL's simple and realistic simulators give different executed amounts, because only the realistic one enforces lot units — [rl_order_execution README](https://raw.githubusercontent.com/microsoft/qlib/main/examples/rl_order_execution/README.md)

### Inferences
These are design recommendations derived from the findings above; they are not cited facts.
1. **Declarative strategy spec, not code.** The agent submits only one of:
   - factor expressions in a restricted DSL (as in Qlib's restricted parser), or
   - a score DataFrame of `(datetime, instrument) -> score`.

   The engine owns signal→order translation (e.g. TopkDropout-like with fixed `topk`, `n_drop`, `hold_thresh`), fills, costs and metrics.
2. **Process and filesystem isolation.**
   - Run agent-generated code (if any) in a separate process or container with read-only access to the training-window data only. It gets no network, no write access to the engine, the data or the results directory, and the evaluation-period data stays physically unavailable.
   - The engine runs in another process. It reads the submitted spec or signal file, validates its schema, and recomputes everything.
   - Hash and pin the engine code, data snapshot and config. Record the hashes with every result.
3. **Frozen config, rejected overrides.**
   - Engine parameters (`deal_price`, the limit rule, `volume_threshold`/`volume_percent`, `open_cost`/`close_cost`/`min_cost`, `trade_unit`, matching type) come from a sealed config.
   - Any attempt to pass them through the spec fails. This mirrors Qlib's `ValueError` on unknown kwargs and avoids RQAlpha's "strategy code wins" priority.
4. **Time-causality enforced by the engine.**
   - Signals timestamped at bar t can only fill at t+1 (open or VWAP), following RQAlpha `next_bar` and the Nautilus look-ahead rule.
   - Labels and features are computed by the engine from point-in-time data.
   - The engine checks that a signal at t is unchanged when future data is truncated, i.e. recompute on data cut at t.
5. **A-share rules in the engine, not the strategy:**
   - T+1 sellable quantity, per-bar limit-up/limit-down flags (Qlib tuple-expression style), suspension, zero-volume bars (`inactive_limit`), lot rounding, stamp duty on sells, minimum commission and a participation cap.
6. **Fixed metric set and holdout.**
   - Report IC/RankIC (SigAna style), portfolio return net of cost, turnover, and execution indicators (FFR, PA, POS), all computed by the engine.
   - Keep a sealed final test segment the agent never sees scores for.
   - Use deterministic seeds.
7. **Cross-checks for tamper and bug detection.**
   - Two independent engines on the same signal, e.g. a vectorized fast path plus an event-driven realistic path, should agree within tolerance, as Qlib's `check_diff_freq` and the simple/realistic simulator comparison do.

### Gaps
- I found no published framework documentation that specifically addresses adversarial or LLM-agent tampering of backtests. The recommendations above are inferred from general security patterns and the framework behaviors cited, not from a dedicated source.
