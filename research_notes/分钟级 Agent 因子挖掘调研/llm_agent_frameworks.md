# LLM-Agent (and related) Automated Factor / Quant-Strategy Mining Frameworks — Workflows, Action Spaces, Evaluation, Anti-Overfitting

Research date: 2026-11-02. Sources are primary (arXiv full text, GitHub repo files/API) unless noted. "Open" = what the agent/search may change; "Fixed" = held constant by the harness.

## Q1. Loop stages and action space (what is open vs fixed) per framework

### Takeaway
Almost every LLM factor-mining system published 2023–2026 opens only the **factor definition** (an operator-DSL expression or a Python function over daily OHLCV(+VWAP)) and freezes everything downstream: label (next-day / 5-day / 10-day return), combiner model (almost always LightGBM), portfolio rule (Qlib `TopkDropoutStrategy` top-50 / drop-5), costs (0.05% buy / 0.15% sell, min 5 CNY), and splits. RD-Agent(Q) is the only major system that also opens the **model architecture + training hyper-parameters**; FaVOR is the only one that opens **entry-rule thresholds** (tuned with Optuna on validation). None lets the agent change the label/horizon, portfolio construction, execution, or risk control. Newer work (AlphaSchema, "Propose, Don't Judge") deliberately *moves search control out of the LLM* (schema planner / frozen statistical referee).

### Cited Findings

**Comparison table** (details and sources in bullets below)

| Framework (date, venue) | Loop stages | Open dimensions (agent may change) | Fixed parts | Freq / market / target | Evaluation & backtest | Anti-overfit / leakage | Open source |
|---|---|---|---|---|---|---|---|
| RD-Agent(Q) / R&D-Agent-Quant (arXiv 2505.15155, May 2025; NeurIPS 2025) | Specification → Synthesis (hypothesis, "knowledge forest") → Implementation (Co-STEER code agent) → Validation (Qlib) → Analysis/feedback; contextual Thompson-sampling bandit picks factor-vs-model each loop | Python factor code (writes `result.h5`); PyTorch model class (`model_cls`, Tabular or TimeSeries) + model hyper-params + training hyper-params (n_epochs, lr, early_stop, batch_size, weight_decay) | Label `Ref($close,-2)/Ref($close,-1)-1`; LightGBM with fixed params in factor loop; Alpha158/Alpha20 base features; TopkDropout topk=50 n_drop=5; costs; splits; RobustZScoreNorm/CSZScoreNorm | Daily; CSI300 (main), CSI500, NASDAQ100; next-day return | IC/ICIR/RankIC/ARR/IR/MDD; Qlib backtest 2017-01-01→2020-08-01 (CSI300) | IC-correlation dedup (≥0.99 vs SOTA dropped); validation split; extended test 2024-01→2025-06 post-cutoff | Yes, MIT, github.com/microsoft/RD-Agent (~15.2k★, pushed 2026-10-21) |
| AlphaAgent (arXiv 2502.16789, Feb 2025; KDD'25) | Idea agent (hypothesis) → Factor agent (expression+description) → Eval agent (backtest) → feedback | Factor expression over operator library (AST), window params, NL description | LightGBM (max depth 4), TopkDropout 50/5, costs, OHLCV only, label | Daily; CSI500, S&P500; next-day return | Qlib; train 2015–2019, valid 2020, test 2021–2024/25 | Originality (AST subtree similarity vs Alpha101 zoo), hypothesis–factor alignment (LLM-scored), complexity (length, free params, #features) | Code public, github.com/RndmVariableQ/AlphaAgent (428★); no LICENSE file detected by GitHub API |
| QuantaAlpha (arXiv 2602.07085, Feb 2026, v3 May 2026) | Hypothesis → factor expression (AST) → compiled code (LLM repair) → consistency verifier → backtest → trajectory mutation (rewrite worst step) / crossover (splice high-reward segments) | Hypothesis, factor expression, code repair; trajectory steps | LightGBM, TopkDropout 50/5 equal weight, costs 0.05/0.15%, next-day-open execution, 9.5% limit, label `P(t+2)/P(t+1)-1` | Daily; CSI300 (main), CSI500, S&P500 | Qlib; train 2016–2020, valid 2021, test 2022-01-01→2025-12-26 | Complexity score (length ≤250, ≤6 base features), AST redundancy vs zoo, output-correlation filter |ρ|>0.7 on 2021 valid; test never touched | Yes, github.com/QuantaAlpha/QuantaAlpha (~1.6k★); README shows MIT badge, API detects no licence file |
| CogAlpha (arXiv 2511.18850, Nov 2025; ACL 2026) | 21 task agents in 7-level hierarchy → Python factor code → multi-agent quality checker (code quality, repair, judge, logic-improve) → fitness filter → "thinking evolution" (mutation/crossover), 24 generations | Python factor function (per-stock DataFrame → Series) | LightGBM, 10-day forward return label, TopK 50 / ≤5 swaps, costs | Daily; CSI300; 10-day return | Qlib; train 2011–2019, valid 2020, test 2021-01→2024-12 | Leakage unit test + Judge agent, NaN>30% drop, percentile fitness thresholds | Code "will be released" (not found) |
| FactorMiner (arXiv 2602.14670, Feb 2026; ICLR 2026 page) | "Ralph Loop": retrieve memory → generate → multi-stage evaluate → library update → distill memory (success patterns + forbidden regions) | Formula over 60+ typed operators; leaves $open,$high,$low,$close,$volume,$amt,$vwap,$returns | Operator library, fields, admission protocol | **10-minute bars**; CSI500/CSI1000/HS300 + 64 Binance crypto; next 10-min open-to-close return | IC/ICIR; admit IC≥0.04 & corr<0.5; train 2024, test 2025; quintile L/S (no costs in main) | Correlation admission, frozen Top-40 OOS, forbidden-region memory | No code URL in paper |
| Navigating the Alpha Jungle (LLM+MCTS, arXiv 2505.11122, v3 Nov 2025) | MCTS: select node → sample weak dimension (Effectiveness/Stability/Turnover/Diversity/Overfitting) → LLM refinement suggestion → new formula → backtest → backprop | Formula + 3 parameter sets per formula | Downstream LightGBM/MLP, top-k/drop-n backtest | Daily; CSI300, CSI1000; 10-day & 30-day returns | train 2011–2020, test 2021-01→2024-11; cost 0.15% | Frequent Subtree Avoidance; leakage probe (App. H) | No URL found |
| AlphaSchema (arXiv 2607.26642, Jul 2026) | Sample schema plans in semantic space → quota scheduler (explore / surrogate exploit / mutate) → LLM realizes plan as code → validation (numerics, leakage) → backtest → retrain LightGBM surrogate | LLM: only plan→code realization (operators, form, numeric params); **search over schema (event, context, qualities, direction, output)** done by surrogate | Label 5-day close-to-close; LightGBM combiner; Top50/Drop5; costs | Daily; CSI300 (+CSI500 appendix) | train 2016–2020, valid 2021–2022, test 2023–2025 | Leakage screen; reward = RankIC+RankICIR+lag-sensitivity term | github.com/JingyangYi/AlphaSchema |
| FaVOR (arXiv 2608.30192, ~Sep 2026) | Stage 1 hypothesis → observable conditions → candidate formulas; Stage 2 LLM factor-level validation vs conditions (no future returns); Stage 3 integration: combinations + directional-selectivity sweep; thresholds tuned (Optuna, Calmar) on validation | Hypothesis, formulas (predefined operators), **entry thresholds per factor** | Execution protocol, exit rules (holding horizon T / stops), costs | Daily (Qlib); CSI500, S&P500 | train 2022–2023, valid 2024, test 2025 | Validation without returns; thresholds frozen before test | github.com/damilab/FaVOR (per abstract listing) |
| "Propose, Don't Judge" (arXiv 2609.27051, Sep 2026) | LLM controller proposes factor families/submissions + diagnostics; frozen betting referee admits (online e-BH) / retires (e-detector) | Which families, submission allocation, retirement diagnoses, self-written probes | "Trust kernel": referee, library within epoch, point-in-time data, append-only log, cost meter | Daily rank-IC; CSI500 2015-01→2026-08 | e-process on post-submission daily rank-IC | Anytime-valid FDR; leaky-referee comparison | Code section not read |
| Alpha-GPT 1.0/2.0 (arXiv 2308.00016 Jul 2023 / 2402.09746 Feb 2024) | Ideation (idea polisher + retrieval) → LLM seed formulas → GP search → analyst backtest summary → human feedback | Formula seeds; GP evolves them; human steers | Backtest engine, operators | Intraday OHLCV/VWAP (CN, US); WorldQuant IQC | IC, Sharpe, turnover | Human-in-the-loop | No code release found |
| QuantAgent (arXiv 2402.03755, Feb 2024) | Inner loop (writer↔judge with knowledge base) / outer loop (real-market feedback updates KB) | Trading signals / factors | — | Not specified on abstract page | — | — | No repo found |
| AlphaForge (arXiv 2406.18394, AAAI 2025 per common citation) | Generator (noise→RPN formula) trained against learned fitness predictor; daily dynamic combination | Formula (non-LLM generative) | Combiner = rolling linear regression; Top50/≤5 changes | Daily; CSI300/500; ~20-day VWAP return | Qlib; test 2018–2022 rolling | Diversity loss; IC/ICIR filters | github.com/DulyHao/AlphaForge (444★) |
| alphagen (arXiv 2306.12964, KDD'23) / AlphaQCM (ICML 2025) / AlphaSAGE (arXiv 2509.25055, ICLR 2026) | RL (alphagen, AlphaQCM: distributional RL) or GFlowNet (AlphaSAGE) token-by-token formula generation; reward from combination IC | Formula tokens over fixed operators + 6 features | Operators, features, combiner | Daily; CSI300/500/1000, S&P500 | IC/ICIR/AR/MDD/SR | Novelty/structure reward, entropy reg. | AlphaSAGE github.com/BerkinChen/AlphaSAGE; alphagen github.com/ICT-FinD-Lab/alphagen (~1.3k★) |
| TradingAgents (arXiv 2412.20138, Dec 2024) | Analysts → bull/bear debate → trader → risk team | Per-asset trading decisions (not factor mining) | — | Daily, single tickers | Cum. return, Sharpe, MDD | — | Apache-2.0, github.com/TauricResearch/TradingAgents (~120k★) |

**RD-Agent(Q) — details**
- Five units: Specification, Synthesis, Implementation (Co-STEER), Validation, Analysis; a two-armed contextual Thompson-sampling bandit with an 8-dim performance state chooses "factor" or "model" optimisation each loop — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- In R&D-Factor the predictor is fixed to LightGBM; in R&D-Model the input factor set is fixed to Alpha 20; the LLM sees schema-level information only, not raw market data — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- Synthesis keeps a "knowledge forest" of past hypotheses/feedback; Co-STEER keeps a task-code-feedback knowledge base retrieved by embedding similarity (text-embedding-ada-002) — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- Repo template `conf_combined_factors_sota_model.yaml`: market csi300, benchmark SH000300; label `Ref($close, -2)/Ref($close, -1) - 1`; features = Alpha158DL expressions + `combined_factors_df.parquet` (agent factors); RobustZScoreNorm + Fillna on features, DropnaLabel + CSZScoreNorm on label; `TopkDropoutStrategy topk: 50, n_drop: 5`; `limit_threshold: 0.095, deal_price: close, open_cost: 0.0005, close_cost: 0.0015, min_cost: 5`; default segments train 2008-01-01→2014-12-31, valid 2015-01-01→2016-12-31, test 2017-01-01→; model `GeneralPTNN` with Jinja-templated `n_epochs, lr, early_stop, batch_size, weight_decay, num_features, num_timesteps` — [RD-Agent factor_template](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml)
- In the factor-only loop (`conf_combined_factors.yaml`) the model is `LGBModel` with hard-coded params (learning_rate 0.2, max_depth 8, num_leaves 210, lambda_l1 205.6999, lambda_l2 580.9768, colsample 0.8879, subsample 0.8789) — [RD-Agent conf_combined_factors.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml)
- Factor interface: agent writes a Python file with `calculate_{name}` and saves `result.h5` indexed by (datetime, instrument); "One factor should statically define one output… last 10 days momentum and last 20 days momentum should be two different factors"; "Each number in the factor represents a physics value to an instrument on a day" — [RD-Agent experiment/prompts.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/experiment/prompts.yaml)
- Model proposal JSON includes `hyperparameters`, `training_hyperparameters` (n_epochs, lr, early_stop — "you CAN CHANGE"), `model_type` Tabular/TimeSeries; prompt: "Avoid re-implementing previous factors as those that surpassed SOTA are already included in the factor library" — [RD-Agent qlib/prompts.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/prompts.yaml)
- CLI scenarios: `rdagent fin_quant` (joint), `fin_factor`, `fin_model`, `fin_factor_report` (extract factors from financial reports) — [RD-Agent README](https://github.com/microsoft/RD-Agent)

**AlphaAgent — details**
- Idea Agent (CoT hypothesis: observations, knowledge, justification, specification) → Factor Agent → Eval Agent; factors are ASTs over operators (e.g., TS_MIN, SMA) on $open/$high/$low/$close/$volume with LLM-chosen window sizes — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)
- Regularisation: originality = largest common subtree vs alpha zoo (Alpha101); hypothesis alignment LLM-scored (α=0.5); complexity = symbolic length + free params + log #features — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)

**QuantaAlpha — details**
- Trajectory-level evolution: mutation rewrites the lowest-reward step with prefix frozen; crossover recombines high-reward segments; budget 10 directions × 3 expressions × 5 iterations, ~1.8M tokens, ~20 h per run — [arXiv 2602.07085](https://arxiv.org/html/2602.07085)
- Mining uses a precomputed `daily_pv.h5` plus Qlib cn_data 2016–2025 (HF dataset QuantaAlpha/qlib_csi300) — i.e., inherits RD-Agent's factor-data interface — [QuantaAlpha README](https://github.com/QuantaAlpha/QuantaAlpha)

**CogAlpha — details**
- 7 levels / 21 agents (e.g., L3 Liquidity, OrderImbalance; L6 RegimeGating); initial pool 80, parent pool 32, children 96; 24 generations in 3 sub-cycles; adaptive generation feeds two best valid + two worst invalid alphas with analyses; five guidance paraphrase styles — [arXiv 2511.18850](https://arxiv.org/html/2511.18850v1)

**FactorMiner — details**
- Memory stores "successful patterns" and "forbidden regions" (factor families highly correlated with library); replacement rule when IC≥0.10, ≥1.3× correlated factor's IC and exactly one conflict; 110 admitted A-share factors, avg off-diagonal |ρ| 0.203 — [arXiv 2602.14670](https://arxiv.org/html/2602.14670v1); summary in [search/ICLR listing](https://iclr.cc/virtual/2026/10021261)

**AlphaSchema — details**
- A schema plan p=(event, context, qualities Q, direction d, output o); LLM "independently chooses the operators, functional form, and numerical parameters"; feedback is attached to plans not code; a LightGBM reward model over schema features is retrained on cumulative plan–reward data; authors report implementations of the same plan by different LLMs have comparable predictive quality ("stronger LLMs mainly improve implementation success rate") — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1); secondary summary [besthub](https://www.besthub.dev/articles/alphaschema-semantic-space-exploration-reshapes-llm-alpha-mining-31cf6b3aabae)
- Default code-agent backend DeepSeek-V4-Flash — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1)

**FaVOR — details**
- Stage 2 validation "uses market variables observed at time t rather than future returns at t+T"; Stage 3 signal fires only when every factor in a combination crosses its in-sample quantile; factor-specific thresholds optimised with Optuna on Calmar ratio over the validation set, then frozen; four LLM agents, GPT-4o default — [arXiv 2608.30192](https://arxiv.org/pdf/2608.30192)

**"Propose, Don't Judge" — details**
- Agent's "surface may evolve without limit" (families, allocation, retirement diagnoses, self-written probes that never enter admission tests); frozen "trust kernel" = referee, library within an epoch, point-in-time data layer, append-only decision log, cost meter — [arXiv 2609.27051](https://arxiv.org/html/2609.27051v1)

**Alpha-GPT / QuantAgent**
- Alpha-GPT: Trading Idea Polisher → Quant Developer LLM seed formulas → Alpha Compute Framework (genetic programming) → Analyst agent backtest summaries; Llama-3-70B + BGE-M3; intraday OHLCV/VWAP for CN/US stocks, 19 basic operators — [arXiv 2308.00016v2](https://arxiv.org/html/2308.00016v2)
- Alpha-GPT 2.0 extends human-in-the-loop to modelling and analysis phases (abstract only, no metrics) — [arXiv 2402.09746](https://arxiv.org/abs/2402.09746)
- QuantAgent: inner loop refines via knowledge base, outer loop tests in real-world scenarios and updates KB — [arXiv 2402.03755](https://arxiv.org/abs/2402.03755)

**Navigating the Alpha Jungle (MCTS)**
- Dimension sampled by softmax favouring weakest; LLM writes refinement suggestion then formula plus "alpha portrait"; 3 symbolic parameter sets per formula; FSA forbids top-3 frequent closed root subtrees — [arXiv 2505.11122](https://arxiv.org/html/2505.11122v3)

**Related systems not deeply reviewed**
- EvoAlpha (NeurIPS 2025 GenAI-in-Finance workshop): evolutionary prompts with backtest feedback, used for sparse ℓ0 portfolio optimisation — [NeurIPS page](https://neurips.cc/virtual/2025/132561)
- AlphaAgentEvo (ICLR 2026): agentic RL, 4B model outperforming LLM-evolution with closed reasoning models — [ICLR page](https://iclr.cc/virtual/2026/poster/10007685)
- QuantEvolver (arXiv 2605.15412): RL fine-tuned LLM self-evolving factor discovery — [emergentmind](https://www.emergentmind.com/papers/2605.15412)
- Chain-of-Alpha (arXiv 2508.06312), AlphaBench (ICLR 2026 benchmark), FactorEngine (arXiv 2603.16365, program-level factors), Alpha-R1 (arXiv 2512.23515) — [Chain-of-Alpha](https://arxiv.org/pdf/2508.06312); [AlphaBench](https://iclr.cc/virtual/2026/poster/10008434); FactorEngine mention via [search result](https://arxiv.org/html/2602.14670v1)
- AlphaEvolve (Cui et al. 2021, arXiv 2103.16196) is a non-LLM AutoML evolutionary alpha method with correlation pruning; DeepMind's AlphaEvolve (2025) is a general coding agent — [ar5iv 2103.16196](https://ar5iv.labs.arxiv.org/html/2103.16196)

### Inferences
- The de-facto "frozen harness" across the literature is RD-Agent's Qlib template (label t+1→t+2 close, LightGBM, Top50/Drop5, 0.05%/0.15%/5 CNY, 9.5% limit). QuantaAlpha, AlphaAgent, CogAlpha and AlphaSchema all reuse it nearly verbatim, so their numbers are comparable only to each other, and none tests whether the factor survives a different label/portfolio rule.
- For a minute-level A-share design, the open dimensions in this literature map cleanly to: (a) factor expression/code (universal), (b) model + training HPs (RD-Agent only), (c) signal thresholds/entry rule (FaVOR only). Label/horizon, execution and risk control are an unexplored open dimension — and opening them substantially enlarges the multiple-testing problem.
- Trend 2026: decouple "what to search" from the LLM (AlphaSchema surrogate over semantics; frozen referee) because LLM-driven search is hard to reproduce and LLM choice mainly affects implementation success, not factor quality.

### Gaps
- RD-Agent(Q) paper does not say explicitly whether training HPs are agent-tuned in the paper's runs (the repo prompts allow it).
- Alpha-GPT/QuantAgent full experimental setups and code: not found.
- AlphaPilot: no paper by that name found.
- AlphaQCM's own arXiv page not fetched; details only via AlphaSAGE's description and ICML page.

## Q2. Data frequency, markets, predicted target

### Takeaway
Daily bars dominate (RD-Agent(Q), AlphaAgent, QuantaAlpha, CogAlpha, AlphaSchema, FaVOR, MCTS, AlphaForge, RL baselines). Only **FactorMiner** runs on intraday data (10-minute bars, A-share CSI500/CSI1000/HS300 + crypto, predicting next 10-minute open-to-close return); Alpha-GPT mentions intraday OHLCV but gives no systematic setup. No 1-minute or tick-level LLM-agent factor-mining system was found. Markets: CSI300 is the default benchmark; CSI500, CSI1000, S&P500, NASDAQ100 secondary.

### Cited Findings
- RD-Agent(Q): daily, CSI300 (300 large caps), extended CSI500 and NASDAQ100; label next-day return τ=1 cross-sectionally z-scored — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- AlphaAgent: daily CSI500 (Baostock) and S&P500 (Yahoo); only open/high/low/close/volume + 4 base alphas; label next-day return — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)
- QuantaAlpha: daily, open/high/low/close/volume/vwap; CSRankNorm on features and labels; label y_t = P_{t+2}^close/P_{t+1}^close − 1 — [arXiv 2602.07085](https://arxiv.org/html/2602.07085)
- CogAlpha: daily OHLCV CSI300 via Qlib; label 10-day forward return, trades at open — [arXiv 2511.18850](https://arxiv.org/html/2511.18850v1)
- FactorMiner: 10-minute bars, CSI500/CSI1000/HS300 (>25M data points) + 64 Binance crypto; target next 10-minute open-to-close price change; train 2024, test 2025; Gemini 3.0 Flash — [arXiv 2602.14670](https://arxiv.org/html/2602.14670v1)
- AlphaSchema: CSI300 daily, 5-day forward close-to-close return; train 2016–2020, valid 2021–2022, test 2023–2025 — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1)
- FaVOR: CSI500 (Baostock) and S&P500 (Yahoo), Qlib; train 2022–2023, valid 2024, test 2025 — [arXiv 2608.30192](https://arxiv.org/pdf/2608.30192)
- MCTS (Alpha Jungle): CSI300 & CSI1000 daily; 10- and 30-day returns; trades at close — [arXiv 2505.11122](https://arxiv.org/html/2505.11122v3)
- AlphaForge: CSI300/CSI500; label `Ref(VWAP,-21)/Ref(VWAP,-1)-1` (~20-day) — [arXiv 2406.18394](https://arxiv.org/html/2406.18394)
- AlphaSAGE: CSI300/500/1000 + S&P500 daily (K=252) — [arXiv 2509.25055](https://arxiv.org/html/2509.25055)
- Hubble (arXiv 2604.09601, Mar 2026): daily, S&P500 (501 stocks), discovery 2022-01→2025-05, OOS 2025-06→2026-03-13 — [arXiv 2604.09601](https://arxiv.org/html/2604.09601)
- Alpha-GPT: intraday OHLCV, VWAP, sector data for CN & US; reached ~top 5–10% in a high-frequency competition — [arXiv 2308.00016v2](https://arxiv.org/html/2308.00016v2)

### Inferences
- A minute-level A-share agent would be essentially novel territory; FactorMiner (10-min, IC threshold 0.04, corr 0.5) is the closest reference design and its GPU/fast-operator engine (≈6 min per 1,000 factors vs ≈70 with Pandas) indicates that evaluation throughput, not LLM cost, is the bottleneck at intraday frequency.
- The daily-label convention (signal at close t, trade at t+1, label t+1→t+2) needs explicit redesign at minute frequency (bar-close signal, next-bar execution, T+1 sell constraint in A-shares).

### Gaps
- No primary source found for any LLM-agent system using 1-minute bars or Level-2/tick data for factor mining.

## Q3. Evaluation, backtest setup, headline results

### Takeaway
Evaluation is almost uniformly: factor-level IC/RankIC/ICIR → LightGBM combiner → Qlib TopkDropout (50/5) backtest with 0.05%/0.15% costs → ARR/IR/MDD. Headline numbers are not comparable across papers (different test windows, universes, labels). Reported CSI300 test IC ranges ~0.04–0.06; ARR ranges from ~4% (QuantaAlpha 2022–2025) to ~14% (RD-Agent(Q) 2017–2020).

### Cited Findings
- RD-Agent(Q) CSI300 test (2017-01→2020-08), o3-mini: IC 0.0532, ICIR 0.4278, RankIC 0.0495, ARR 14.21%, IR 1.7382, MDD −7.42%; baselines Alpha158 ARR 5.70%, MASTER ARR 8.96%; cost "under $10" per run; bandit got 44 loops vs 33 for alternatives in ~12 h (not equal loop counts) — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- RD-Agent(Q) backtest: top 50 / drop 5, signal at close t, 0.095 limit, buy 0.05% / sell 0.15% / min 5 CNY; NASDAQ100 top-20, 0.1% cost; appendix D.1 also says "0.5% in CSI 300" (internal inconsistency) — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- AlphaAgent (2021–2024): CSI500 IC 0.0212, ICIR 0.1938, AR 11.00%, IR 1.488, MDD −9.36%; S&P500 IC 0.0056, AR 8.74%; base LLM GPT-3.5-turbo; test period stated inconsistently (2021-01→2025-01 vs 2021–2024) — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)
- QuantaAlpha CSI300 (2022–2025), GPT-5.2: IC 0.0472, RankIC 0.0459, IR 0.6453, ARR 4.68%, MDD 11.80%; DeepSeek-V3.2 IC 0.0461; Claude-4.5-Sonnet IC 0.0445; strategy uses ~150-factor pool; 966 test days with t-stats — [arXiv 2602.07085](https://arxiv.org/html/2602.07085). Note: search snippets cite "IC 0.1501 with GPT-5.2" which does not match the fetched table — [arXiv abs](https://arxiv.org/abs/2602.07085) vs [HTML](https://arxiv.org/html/2602.07085)
- CogAlpha CSI300 (2021–2024): IC 0.0591, RankIC 0.0814, ICIR 0.3410, AER 16.39%, IR 1.8999; gpt-oss-120b — [arXiv 2511.18850](https://arxiv.org/html/2511.18850v1)
- FactorMiner CSI500 2025: Top-40 library IC 8.25%, ICIR 0.77 vs AlphaAgent 5.90%/0.46 (same engine); crypto IC 3.82%; costs only stress-tested 1–11 bps — [arXiv 2602.14670](https://arxiv.org/html/2602.14670v1)
- AlphaSchema CSI300 (2023–2025): OHLCV pool (120 factors) IC 0.0382, ICIR 0.2374, RankIC 0.0498, RankICIR 0.2912, IR 0.7624, AER 8.53%; +Fundamental (150 factors) IR 1.0877, AER 11.94%; Top50/Drop5, deal_price=open next day, 0.05%/0.15%/min 5, limit 0.095 — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1)
- FaVOR 2025 OOS after costs: CSI500 AR 0.2067, IR 1.5295, MDD −0.0853; S&P500 IR 1.1315; positive across GPT-4o, GPT-5.4-mini, Gemini-2.5-Flash, Claude-Sonnet-4.6, Llama-3.3-70B, Qwen3-235B — [arXiv 2608.30192](https://arxiv.org/pdf/2608.30192)
- MCTS: LightGBM IC 0.0549, IR 1.1792; Gemini-2.0-flash-lite IR 1.27 at ~$7.5 vs GPT-4.1 IR 1.23 at ~$74.4; k = top 10% of pool, n = k/horizon, 0.15% cost — [arXiv 2505.11122](https://arxiv.org/html/2505.11122v3)
- AlphaSAGE CSI300 test 2022–2024: IC 0.079, AR 7.62%, SR 1.71 vs AlphaGen IC 0.058, AlphaQCM 0.043, AlphaForge 0.041, AlphaAgent 0.051 — [arXiv 2509.25055](https://arxiv.org/html/2509.25055)
- AlphaForge: CSI300 IC 4.40%, RankIC 5.89%; ~9 months live on 3M RMB CSI500 account, 21.68% excess — [arXiv 2406.18394](https://arxiv.org/html/2406.18394)
- Alpha-GPT: WorldQuant IQC 2024 top-10 worldwide; mean IC 0.58% (seed) → 1.23% (GP) → 2.23% (one interaction round + search) — [arXiv 2308.00016v2](https://arxiv.org/html/2308.00016v2)
- Hubble: 3 rounds × batch 20, 104 valid evaluations; OOS Range-1/Range-2 IC HAC t-stats 2.98/3.01; Trend-1 IC turned negative; no cost-aware backtest — [arXiv 2604.09601](https://arxiv.org/html/2604.09601)

### Inferences
- Many "SOTA" claims are within-harness and partially conflict (e.g., AlphaAgent IC 0.051 on CSI300 in AlphaSAGE's table vs 0.0212 on CSI500 in its own paper) — cross-paper rankings should not be trusted; re-run baselines in one's own harness (as FactorMiner did).
- Daily-rebalanced Top50/Drop5 with ~0.2% round-trip is the cost model everyone uses; for minute-level, cost/impact and turnover must be modelled explicitly because FactorMiner's main results are pre-cost.

### Gaps
- No paper reports a minute-level portfolio backtest with realistic A-share T+1 and intraday costs.

## Q4. Anti-overfitting and anti-LLM-leakage mechanisms

### Takeaway
Three families: (1) **structural regularisers** — AST originality vs a factor zoo, complexity caps, frequent-subtree bans, correlation admission gates, forbidden-region memory; (2) **protocol hygiene** — chronological train/valid/test, test untouched during search, frozen OOS selection, thresholds frozen before test; (3) **statistical gating** — only "Propose, Don't Judge" uses formal multiple-testing control (anytime-valid e-BH on post-submission data). LLM pretraining leakage is mostly asserted away (post-cutoff test windows; one probe study) rather than controlled; Hubble and "Propose, Don't Judge" explicitly flag it as unresolved. No system reports explicit trial counting / deflated Sharpe for the agent's own search except "Propose, Don't Judge".

### Cited Findings
- RD-Agent(Q): new factor excluded if max time-averaged IC with SOTA factors ≥0.99; Co-STEER "health-check" for leakage-free factors; argues LLM never sees temporal splits; extended test 2024-01→2025-06 after GPT-4o cutoff (Oct 2023) and "almost entirely" after o4-mini cutoff (Jun 2024) — [arXiv 2505.15155](https://arxiv.org/html/2505.15155)
- AlphaAgent: originality/alignment/complexity regularisation; no discussion of pretraining leakage; notes LLMs tend to replicate well-known factors (RSI, momentum) which worsens crowding — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)
- QuantaAlpha: complexity limits (≤250 symbols, ≤6 features; App. E uses ≤200/≤4 — inconsistent), AST dedup, |ρ|>0.7 output-correlation filter on 2021 validation; test 2022–2025 never accessed; no pretraining-overlap discussion — [arXiv 2602.07085](https://arxiv.org/html/2602.07085)
- CogAlpha: leakage unit test + Judge agent leakage rejection; NaN>30% discard — [arXiv 2511.18850](https://arxiv.org/html/2511.18850v1)
- FactorMiner: IC≥0.04 & |ρ|<0.5 admission; Top-40 chosen on 2024 then frozen for 2025 — [arXiv 2602.14670](https://arxiv.org/html/2602.14670v1)
- MCTS Alpha Jungle App. H: prompted GPT-4.1/Gemini/DeepSeek "known strong alphas" ≈ random baseline (IC ~0.012–0.013 vs 0.0126) → authors conclude no leaked superior-alpha knowledge — [arXiv 2505.11122](https://arxiv.org/html/2505.11122v3)
- Hubble: exec-free AST sandbox (whitelist nodes, depth/node limits, DSL semantic check); crowding/similarity/family-concentration penalties, family cap 2; states LLM pretraining may include post-in-sample literature and "the DSL cannot prevent this"; flags narrative bias of LLM rationales — [arXiv 2604.09601](https://arxiv.org/html/2604.09601)
- AlphaSchema: execution, numerical-stability and look-ahead leakage screens before backtest — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1)
- FaVOR: validation stage uses only time-t observables (no returns); thresholds optimised on 2024 validation, frozen for 2025 — [arXiv 2608.30192](https://arxiv.org/pdf/2608.30192)
- "Propose, Don't Judge": frozen referee admitted 11.7 false factors per campaign vs 86–196 for leaky referees (script) — 5–11× fewer; cost: median ~500 trading days to admit a true factor; certified book net Sharpe +0.33–0.50 vs ungated +0.59–0.77; guarantee requires proposer to have no information on post-submission outcomes, so LLM historical replays (training covering campaign years) are not instances of the guarantee — [arXiv 2609.27051](https://arxiv.org/html/2609.27051v1). Third-party critique argues LLM allocation edge mostly reflects bandit cold start — [GitHub issue #1959](https://github.com/jjakimoto/research-issues/issues/1959)
- AlphaSAGE: novelty reward = 1 − max |IC| vs library; entropy regulariser against mode collapse — [arXiv 2509.25055](https://arxiv.org/html/2509.25055)

### Inferences
- For a minute-level agent with many loops, the protocol should (i) count every evaluated candidate, (ii) gate admission on data the agent never saw (rolling forward / post-submission), and (iii) keep the evaluator, data layer and log outside the agent's write scope — the "trust kernel" pattern.
- Minute-level data partly mitigates LLM leakage (LLMs are unlikely to have memorised intraday microstructure results for specific periods), but published intraday factor families (e.g., from broker research) can still leak; test windows after the model's cutoff remain advisable.

### Gaps
- No framework reports a deflated-Sharpe or Bonferroni-style adjustment for its own number of LLM trials except "Propose, Don't Judge".

## Q5. Open-source status, licence, activity

### Takeaway
Only RD-Agent (MIT, very active), QuantaAlpha, AlphaAgent, AlphaForge, AlphaSAGE, alphagen and TradingAgents (Apache-2.0) have public code; AlphaSchema and FaVOR list repos; CogAlpha, FactorMiner, Hubble, Alpha-GPT, QuantAgent show no released code in the sources checked.

### Cited Findings
- microsoft/RD-Agent: MIT, ~15.2k stars, last push 2026-10-21; NeurIPS 2025 acceptance of R&D-Agent-Quant announced in README — [GitHub API](https://api.github.com/repos/microsoft/RD-Agent); [README](https://github.com/microsoft/RD-Agent)
- RndmVariableQ/AlphaAgent: 428 stars, last push 2026-07-03, no licence file detected by GitHub API — [GitHub API](https://api.github.com/repos/RndmVariableQ/AlphaAgent)
- QuantaAlpha/QuantaAlpha: ~1.6k stars, last push 2026-06-29; README MIT badge but no licence file detected — [GitHub API](https://api.github.com/repos/QuantaAlpha/QuantaAlpha); [README](https://github.com/QuantaAlpha/QuantaAlpha)
- DulyHao/AlphaForge: 444 stars, last push 2024-09-01 — [GitHub API](https://api.github.com/repos/DulyHao/AlphaForge)
- TauricResearch/TradingAgents: Apache-2.0, ~120k stars, last push 2026-10-27 — [GitHub API](https://api.github.com/repos/TauricResearch/TradingAgents)
- ICT-FinD-Lab/alphagen: ~1.3k stars, Apache-2.0 (per GitHub search) — [GitHub search API](https://api.github.com/search/repositories?q=alphagen+in:name)
- AlphaSAGE: github.com/BerkinChen/AlphaSAGE — [arXiv 2509.25055](https://arxiv.org/html/2509.25055)
- AlphaSchema: github.com/JingyangYi/AlphaSchema — [arXiv 2607.26642](https://arxiv.org/abs/2607.26642v1)
- FaVOR: github.com/damilab/FaVOR — [takara tldr](https://tldr.takara.ai/p/2608.30192)
- CogAlpha: code "will be released" — [arXiv 2511.18850](https://arxiv.org/html/2511.18850v1)

### Inferences
- RD-Agent is the only production-grade, maintained base to fork; QuantaAlpha already builds on its `daily_pv.h5` interface, so a minute-level agent could reuse the RD-Agent loop while replacing the Qlib daily template (label, handler, strategy) — those are Jinja-templated YAML, not agent-generated.

### Gaps
- Activity/stars not checked for AlphaSAGE, AlphaSchema, FaVOR repos; licences for AlphaAgent/QuantaAlpha ambiguous.
