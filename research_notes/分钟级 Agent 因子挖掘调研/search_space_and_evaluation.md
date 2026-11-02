# Search Space Beyond Factors & Evaluation Hygiene for Automated / Agent-Driven Quant Pipelines

Researcher notes (compiled 2026-11-02). Scope: (1) which pipeline dimensions automated/AutoML/LLM-agent quant systems search and with what methods; (2) concrete evaluation-hygiene rules (walk-forward, purging/embargo, PBO/CSCV, DSR, trial counting, SPA/multiple testing, cost modeling) and how agent frameworks apply them. Feeds: a fixed "agent-proof" backtester and an "agent vs random search, equal budget" experiment.

Source-quality legend: items marked **[fetched]** were read this session (abstract/HTML). Items marked **[canonical]** are standard references cited from prior knowledge with their canonical URL/DOI; they were not re-fetched this session, so the report writer should treat exact wording/numbers as needing a spot-check. Blog/Substack/DEV posts are flagged as such.

---

## Q1. Which pipeline dimensions are searchable, and which methods/frameworks search them?

### Takeaway
Published automated quant systems overwhelmingly search the **factor (formula) dimension**; a smaller set jointly searches **model/hyperparameters** (RD-Agent(Q) alternates factor vs model via a bandit). Labels/horizon, portfolio construction, execution and risk control are almost always **held fixed** (typically Qlib's TopkDropout topk=50/n_drop=5 with fixed costs) — i.e., they are "open dimensions" in principle but rarely searched in the agent literature, which matters for the design: if they are opened, they add trials and must be counted.

### Cited Findings

**Factor generation (most-searched dimension)**
- Genetic programming over formula trees: gplearn (symbolic regression/transformer in sklearn API) — [gplearn GitHub](https://github.com/trevorstephens/gplearn) **[canonical]**; DEAP (general evolutionary computation) — [DEAP GitHub](https://github.com/DEAP/deap) **[canonical]**. AutoAlpha (GP-based hierarchical evolutionary alpha search, Zhang et al. 2020) — [arXiv:2002.08245](https://arxiv.org/abs/2002.08245) **[canonical]**.
- RL formula generation: AlphaGen (Yu et al., KDD 2023, "Generating Synergistic Formulaic Alpha Collections via RL" — PPO generates RPN token sequences, reward = IC of the *combined* linear factor pool) — [arXiv:2306.12964](https://arxiv.org/abs/2306.12964), [GitHub RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen) **[canonical]**. Alpha² (deep RL for logical formulaic alphas) — [arXiv:2406.16505](https://arxiv.org/abs/2406.16505); RiskMiner (risk-seeking MCTS) — [arXiv:2402.07080](https://arxiv.org/abs/2402.07080) (both listed as related work alongside Alpha-GPT 2.0 [fetched via search listing](https://ideas.repec.org/p/arx/papers/2402.09746.html)).
- Generative + dynamic combination: AlphaForge (AAAI 2025) — generative-predictive network to mine formulaic factors, then time-varying combination weights; reports IC 4.40% on CSI 300 vs 2.55% for next-best formulaic method, beating GP, DSO and RL baselines — [arXiv:2406.18394](https://arxiv.org/html/2406.18394v1) **[fetched snippet]**.
- LLM human-in-the-loop: Alpha-GPT ([arXiv:2308.00016](https://arxiv.org/abs/2308.00016)) and Alpha-GPT 2.0 (human-in-the-loop LLM alpha research) — [arXiv:2402.09746](https://ideas.repec.org/p/arx/papers/2402.09746.html) **[fetched listing]**.
- LLM self-improving strategy generation: QuantAgent ("Seeking Holy Grail in Trading by Self-Improving LLM", Wang et al. 2024, inner loop writer/judge + outer loop real-market feedback) — [arXiv:2402.03755](https://arxiv.org/abs/2402.03755) **[canonical]**.
- LLM evolutionary coding (general-purpose, AlphaEvolve-style): Google DeepMind AlphaEvolve — evolutionary loop of LLM-proposed code edits scored by automated evaluators — [arXiv:2506.13131](https://arxiv.org/abs/2506.13131) **[canonical]**. (I found no peer-reviewed application of AlphaEvolve itself to trading; QuantaAlpha is the closest "trajectory-evolution" analogue.)
- AlphaAgent (Tang et al., Feb 2025): LLM hypothesis→factor→evaluation loop with three regularizers: AST-based **complexity control**, **originality** via AST similarity vs an alpha zoo (e.g. Alpha101), and LLM-scored **hypothesis–expression alignment**. Reported CSI 500 / S&P 500, Jan 2021–Dec 2024: 11.0% annual excess return (IR 1.5) and 8.74% (IR 1.05) after costs; "81% higher hit ratio" and 30% fewer tokens — [arXiv:2502.16789](https://arxiv.org/html/2502.16789v2), code [GitHub RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent) **[fetched snippet]**.
- QuantaAlpha (Feb 2026, v3 May 2026): evolves whole mining **trajectories** with trajectory-level mutation and crossover; main setting 10 parallel planning directions × 5 iterations, 3 expressions per hypothesis; complexity penalty, AST redundancy check vs alpha zoo, LLM consistency check; ~20 h and ~1.8M tokens/run (vs 1.5M AlphaAgent, 2.3M RD-Agent) — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**.
- FactorMiner (Feb 2026): LLM agent with "skills" + **experience memory** storing successful templates and "forbidden regions" (factor families highly correlated with library); "Ralph Loop" retrieve→generate→evaluate→distill; **10-minute bars** on CSI500/CSI1000/HS300 + 64 Binance crypto assets; 60+ operators; target = next-10-min open-to-close return — [arXiv:2602.14670](https://arxiv.org/html/2602.14670v1) **[fetched]**. (Directly relevant: it is the closest published analogue to a minute-level agent factor miner.)

**Model / hyperparameter dimension**
- R&D-Agent(Q) (Microsoft, NeurIPS 2025 D&B): action space = {factor, model}; a two-armed **contextual Thompson-sampling bandit** (8-dim performance state) picks which to improve each loop; Co-STEER code agent implements; inner debugging capped at 10 iterations/task; R&D-Factor and R&D-Model each 6 h, joint 12 h; bandit variant ran 44 loops (24 valid) vs 33 for random/LLM schedulers — [arXiv:2505.15155v2](https://arxiv.org/html/2505.15155v2), code [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) **[fetched]**.
- Generic HPO engines: Optuna (define-by-run, TPE default sampler, pruning) — [arXiv:1907.10902](https://arxiv.org/abs/1907.10902); TPE (Bergstra et al., NeurIPS 2011) — [paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html); Hyperband (successive-halving bandit over budgets) — [arXiv:1603.06560](https://arxiv.org/abs/1603.06560); BOHB — [arXiv:1807.01774](https://arxiv.org/abs/1807.01774); random search as a strong baseline — [Bergstra & Bengio, JMLR 2012](https://jmlr.org/papers/v13/bergstra12a.html) **[canonical]**.

**Label / horizon dimension**
- Triple-barrier labeling and meta-labeling (label = which of profit-take/stop-loss/time barrier is hit first; meta-label decides bet size on a primary signal) — López de Prado, *Advances in Financial Machine Learning* (AFML), Wiley 2018, ch. 3 — [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) **[canonical]**.
- In practice agent frameworks fix the label: FactorMiner fixes next-10-min open-to-close return ([arXiv:2602.14670](https://arxiv.org/html/2602.14670v1)) **[fetched]**; RD-Agent(Q)/QuantaAlpha inherit Qlib daily return labels and do not list the label among actions ([2505.15155](https://arxiv.org/html/2505.15155v2), [2602.07085](https://arxiv.org/html/2602.07085)) **[fetched]**.

**Portfolio construction dimension**
- Qlib TopkDropoutStrategy: hold `topk` names, sell `n_drop` worst-ranked each day and replace with best non-held; example `topk=50, n_drop=5` — [Qlib strategy docs](https://qlib.readthedocs.io/en/latest/component/strategy.html) **[fetched]**. QuantaAlpha uses exactly topk=50, n_drop=5, equal weight, next-day open, 9.5% limit threshold; ~10% of holdings replaced per rebalance — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**. n_drop is effectively a **turnover cap / hysteresis** knob (max daily turnover ≈ n_drop/topk = 10%).
- No-trade region / partial adjustment toward an "aim" portfolio (optimal with quadratic costs: trade a fraction toward aim each period) — Gârleanu & Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs", *J. Finance* 2013 — [doi:10.1111/jofi.12080](https://doi.org/10.1111/jofi.12080) **[canonical]**.
- RD-Agent(Q) describes only "a daily long-short trading strategy based on predicted return rankings, with position updates, holding retention rules" — portfolio rules not part of the search — [arXiv:2505.15155v2](https://arxiv.org/html/2505.15155v2) **[fetched]**.

**Execution dimension**
- Optimal execution with temporary/permanent impact (risk-cost trade-off; TWAP is the risk-neutral special case) — Almgren & Chriss, "Optimal execution of portfolio transactions", *J. Risk* 2000/01 — [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) **[canonical]**.
- RL order execution inside Qlib (Oracle Policy Distillation) — [arXiv:2103.10860](https://arxiv.org/abs/2103.10860) **[canonical]**; FinRL (DRL library for trading strategies / portfolio allocation; agent searches the policy end-to-end) — [arXiv:2011.09607](https://arxiv.org/abs/2011.09607), [GitHub AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) **[canonical]**.

**Risk-control dimension**
- Volatility targeting / volatility-managed portfolios (scale exposure by c/σ²_{t-1}) — Moreira & Muir, *J. Finance* 2017 — [doi:10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513) **[canonical]**.
- Stop-loss rules (conditions under which they add value depend on return serial correlation) — Kaminski & Lo, *J. Financial Markets* 2014 — [doi:10.1016/j.finmar.2013.07.001](https://doi.org/10.1016/j.finmar.2013.07.001) **[canonical]**.

### Inferences
- Taxonomy for the backtester's config space (proposal): **D1 factor/expression** (DSL with operator whitelist), **D2 label** (horizon h ∈ {1,5,10,30,60,…} bars; return type open→close / vwap→vwap; triple-barrier), **D3 model** (linear/LightGBM/MLP + HPO), **D4 portfolio** (top-k or quantile, weighting ∈ {equal, score-proportional, inverse-vol}, n_drop/turnover cap, entry/exit rank buffers, min holding bars), **D5 execution** (fill price ∈ {next open, next-bar VWAP, TWAP over k bars}, participation cap), **D6 risk** (position cap, vol target, gross exposure timing, stop-loss). Every *executed* configuration across D1–D6 is a trial.
- Because no surveyed agent framework opens D4–D6, an agent that does so has more degrees of freedom than published baselines → stronger need for deflation; conversely D5 (execution) and parts of D6 should arguably be **fixed by the harness, not searched**, because searching fill rules is the easiest path to unrealistic backtests (e.g., choosing "fill at close of signal bar").
- Regularizers used by agents as implicit constraints: AST complexity (AlphaAgent/QuantaAlpha symbol length ≤250 and ≤6 base features in QuantaAlpha main setting, ≤200/≤4 in a tighter setting — [2602.07085](https://arxiv.org/html/2602.07085)), originality vs alpha zoo, correlation caps (0.7 QuantaAlpha; 0.5 FactorMiner; IC≥0.99 dedup in RD-Agent). These are reusable as harness-level admission rules.

### Gaps
- Could not verify the exact AlphaGen/AutoAlpha/QuantAgent/AlphaEvolve details this session (cited canonically).
- Found no published "AutoML for the whole quant pipeline" system that jointly searches label + portfolio + execution + risk with an LLM agent; this appears to be an open gap, not a missed source (low confidence — searched only a few queries).
- "AlphaEvolve for alpha" specifically: no reliable source found.

---

## Q2. Typical constraints that keep searched strategies realistic

### Takeaway
Realism constraints in the literature are mostly **Qlib defaults**: 9.5% price-limit threshold (can't buy limit-up / sell limit-down), asymmetric costs 5 bps buy / 15 bps sell, 5 CNY minimum commission, TopK 50 with n_drop 5 (≈10% max daily turnover). Explicit capacity/participation caps are rare in agent papers; FactorMiner (the minute-level one) reports gross results only.

### Cited Findings
- Qlib example exchange kwargs: `limit_threshold=0.095`, `deal_price="close"`, `open_cost=0.0005`, `close_cost=0.0015`, `min_cost=5` — [Qlib strategy docs](https://qlib.readthedocs.io/en/latest/component/strategy.html) **[fetched]**.
- QuantaAlpha: buy 0.05%, sell 0.15% (0.20% round trip), sensitivity at 1.5× and 2.0×; next-day open execution; 9.5% limit threshold — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**.
- FactorMiner: main results **gross**; Appendix K stress tests at 1, 4, 7, 10, 11 bps; transaction-cost-aware backtesting listed as future work — [arXiv:2602.14670](https://arxiv.org/html/2602.14670v1) **[fetched]**.
- RD-Agent(Q): says "realistic transaction costs" in main text; exact numbers in an appendix I could not read in full — [arXiv:2505.15155v2](https://arxiv.org/html/2505.15155v2) **[fetched, partial]**.
- AlphaAgent reports results "after accounting for transaction costs" — [arXiv:2502.16789](https://arxiv.org/html/2502.16789v2) **[fetched snippet]**.
- A-share stamp duty halved from 0.1% to 0.05% effective **28 Aug 2023**, charged on the **seller only** (MoF/STA Announcement No. 39 of 2023) — [gov.cn](https://english.www.gov.cn/policies/policywatch/202308/28/content_WS64ec5513c6d0868f4e8dee23.html), [HKEX circular (Northbound)](https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/SEHK/2023/CT13723B.pdf) **[fetched snippets]**. Implication: Qlib's 15 bps sell cost bakes in the pre-2023 10 bps stamp duty; post-2023 sell-side tax is 5 bps.

### Inferences
- Recommended harness constraints for a minute-level A-share backtester (design proposal, combining the above with market structure):
  - **Tradability masks**: no buy when bar price ≥ limit-up, no sell when ≤ limit-down (use exchange-specific limits by board/ST status, not a single 9.5%); exclude suspended stocks and ST/*ST; exclude stocks listed < N days (e.g. 60 trading days); T+1 (shares bought today cannot be sold today); lot size 100 shares.
  - **Liquidity/capacity**: universe filter by trailing 20-day median turnover value (e.g. top-X by ADV); per-bar fill ≤ participation cap (e.g. 5–10% of that bar's volume); unfilled remainder carried or cancelled deterministically.
  - **Turnover/holding**: cap one-way turnover per rebalance (n_drop/topk style), minimum holding period in bars, and rank hysteresis (enter at rank ≤ k, exit only at rank > k+b).
  - These must be **harness-enforced and not agent-editable**; the agent may only choose parameters inside whitelisted ranges, and each choice counts as a trial.

### Gaps
- Exact current A-share price-limit regimes (±10% main board, ±20% ChiNext/STAR, ±5% ST, ±30% BSE), transfer fee (0.001%) and typical negotiated commission (~1–3 bps with 5 CNY minimum) are standard knowledge but I did not fetch primary exchange sources this session — verify against SSE/SZSE rulebooks before hard-coding.
- No agent paper found that reports capacity (AUM) analysis or participation caps.

---

## Q3. Evaluation discipline: concrete, implementable rules

### Takeaway
Use a three-layer defense: (a) **structural leakage prevention** (point-in-time features, purged/embargoed splits, a locked test set the agent never sees, harness-owned fill rules); (b) **search-aware statistics** computed from a **trial ledger** (DSR with N = number of executed trials or an effective-N from clustering; PBO via CSCV; SPA/StepM vs benchmark; HLZ-style t ≥ 3 hurdles); (c) **single-shot final test** on a post-LLM-cutoff holdout. A 2026 study shows (a) is not redundant: a deliberately leaky oracle with Sharpe 35 passed both DSR and PBO.

### Cited Findings

**Probabilistic & Deflated Sharpe Ratio** — Bailey & López de Prado, "The Deflated Sharpe Ratio", *J. Portfolio Management* 2014 — [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) **[canonical]**; summary [Wikipedia: Deflated Sharpe ratio](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio) **[search result]**.
- PSR (per-period, non-annualized SR̂ over T observations, skew γ₃, *raw* kurtosis γ₄):
  `PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) )`
- Expected max Sharpe of N trials under the null (γ_E ≈ 0.5772 Euler–Mascheroni, V = variance of the trial Sharpe ratios):
  `SR₀ = √V · [ (1−γ_E)·Φ⁻¹(1 − 1/N) + γ_E·Φ⁻¹(1 − 1/(N·e)) ]`
- `DSR = PSR(SR₀)`; accept if DSR ≥ 0.95.
- Practitioner sources emphasize that N is the critical, often-guessed input and recommend agents keep a backtest counter — [DEV Community post (blog-level)](https://dev.to/agentriskdesk/is-your-trading-agents-backtest-overfit-checking-the-deflated-sharpe-ratio-with-one-api-call-1dbd), [halleytech Substack (blog-level)](https://halleytech.substack.com/p/overfitting-purged-cv-deflated-sharpe-quant-research) **[search snippets]**.
- Alpha-R1 (LLM+RL alpha screening) reports DSR under two declared trial counts: an "honestly declared upper bound" covering all variants, grid configs and seeds, and a narrower family count N=34 — [arXiv:2512.23515](https://arxiv.org/pdf/2512.23515) **[search snippet; PDF not readable this session]**.
- VST paper notes its DSR benchmark treats 852 correlated portfolios as independent, so SR₀ is an upper bound (i.e., conservative) — [arXiv:2609.07065](https://arxiv.org/pdf/2609.07065) **[search snippet]**.
- Effective number of trials: cluster trial return series (López de Prado's ONC clustering) and use number of clusters as N, with V computed from cluster-level Sharpes — López de Prado & Lewis, "Detection of false investment strategies using unsupervised learning methods", *Quantitative Finance* 2019 — [SSRN 3167017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017) **[canonical]**.

**Minimum Backtest Length** — Bailey, Borwein, López de Prado & Zhu, "Pseudo-Mathematics and Financial Charlatanism", *Notices of the AMS* 2014 — [SSRN 2308659](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659) **[canonical]**.
- `MinBTL ≈ ( (1−γ_E)·Φ⁻¹(1−1/N) + γ_E·Φ⁻¹(1−1/(N·e)) )² / E[max_N]²  <  2·ln(N) / E[max_N]²` (years, with E[max_N] the target annualized Sharpe). The paper's headline example: with ~5 years of daily data, no more than ~45 independent configurations should be tried before expecting a spurious annualized SR ≈ 1 **[canonical; verify exact number]**.

**Probability of Backtest Overfitting via CSCV** — Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting", *J. Computational Finance* 2017 — [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) **[canonical]**. Algorithm:
1. Build matrix M (T × N): per-period returns of all N trials over the *selection* window.
2. Split rows into S equal contiguous blocks (S even; S=16 → C(16,8)=12,870 combinations).
3. For each combination c of S/2 blocks: J = IS rows, J̄ = OOS rows. Compute performance (e.g. Sharpe) for each trial on J and J̄.
4. n* = argmax IS performance; ω̄_c = relative rank of n* in OOS ∈ (0,1) (rank/(N+1)); λ_c = ln(ω̄_c/(1−ω̄_c)).
5. **PBO = #{λ_c ≤ 0} / #combinations.** Also report the OOS-vs-IS performance-degradation slope and P(OOS Sharpe of n* < 0).
- Reject the search if PBO > ~0.2–0.5 (threshold is a design choice; paper treats PBO near 0.5+ as no skill).
- Python implementations exist (e.g., `pypbo`) **[gap: not verified this session]**.

**Multiple testing for factors** — Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns", *RFS* 29(1) 2016 — [doi:10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059) **[canonical]**:
- A newly discovered factor needs **t > 3.0** (vs 2.0) given the history of factor mining.
- Corrections over M tests at level α: Bonferroni `p ≤ α/M`; Holm (sort ascending, reject p₍ₖ₎ ≤ α/(M+1−k), stop at first failure); BHY FDR `p₍ₖ₎ ≤ k·α/(M·c(M))`, c(M)=Σ_{j=1..M} 1/j.

**Data-snooping tests vs a benchmark** — White, "A Reality Check for Data Snooping", *Econometrica* 2000 — [doi:10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152); Hansen, "A Test for Superior Predictive Ability", *JBES* 2005 — [doi:10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063) **[canonical]**:
- d_{k,t} = perf_t(strategy k) − perf_t(benchmark); H₀: max_k E[d_k] ≤ 0. Statistic: max_k √n·d̄_k/ω̂_k (SPA studentizes; RC does not). p-value from **stationary bootstrap** (Politis–Romano), B ≥ 1,000, mean block length e.g. ~√T or tuned to autocorrelation. SPA re-centres only models with d̄_k ≥ −ω̂_k·√(2·log log n / n) to reduce the influence of poor models. Romano–Wolf StepM identifies *which* strategies beat the benchmark. Implementations: `arch.bootstrap.SPA`, `StepM`, `MCS` — [arch docs](https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparisons.html) **[canonical]**.

**Purged K-fold, embargo, CPCV, walk-forward** — López de Prado, AFML (2018) ch. 7 & 12 — [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) **[canonical]**:
- **Purging**: drop any training sample whose label interval [t₀, t₁] overlaps the test fold's label intervals. For a horizon-h label on minute bars, that is ≥ h bars (plus feature lookback if features use overlapping windows of future-leaking normalizations).
- **Embargo**: additionally drop training samples starting within h_e after each test fold; AFML's rule of thumb h_e ≈ 0.01·T.
- **CPCV**: split into N groups, test on k groups → C(N,k) splits and φ = (k/N)·C(N,k) backtest paths, giving a distribution of OOS Sharpe rather than one path.
- **Walk-forward**: train on [t−L, t), (purge/embargo), test on [t, t+Δ), roll; the only scheme that respects arrow of time but yields a single path → use as final test, CPCV for model selection.

**Leakage is not caught by statistics** — Gençay, "What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-driven trading strategy discovery" (27 Aug 2026) — [arXiv:2608.27734](https://arxiv.org/abs/2608.27734) **[fetched abstract]**:
- Agent acts only via registry-validated tools whose feature space excludes look-ahead; **trial ledger** logs every evaluation and all reported results are deflated by that count; includes transaction costs, impact, borrow.
- Statistics: DSR, PBO, OOS confidence intervals, predicted rank degradation, OOS-collapse checks. Universes: 453-stock point-in-time US equities, 39-ETF multi-asset.
- The deflation threshold grows with the agent's own search **faster than the best in-sample Sharpe**.
- A deliberately leaky oracle (Sharpe 35) **passed both DSR and PBO** → structural guardrails are not redundant with statistics.
- Honest evaluation certified passive benchmarks (OOS CIs excluding zero) but **rejected every LLM-discovered strategy** across two frontier models, budgets up to 100 candidates, and 5 repeated runs.
- A Kinlay blog reports an LLM agent on zero-alpha (random) data achieving in-sample Sharpe above what trial count alone predicts — [jonathankinlay.com (blog, single experiment)](https://jonathankinlay.com/category/backtesting/backtest-overfitting/) **[search snippet]**.

**Fair comparison: agent vs random search**
- Random search is a strong, often under-reported baseline in HPO/NAS; Li & Talwalkar showed random search with early stopping is competitive with NAS methods and stressed reproducibility (seeds, equal compute) — [arXiv:1902.07638](https://arxiv.org/abs/1902.07638); Bergstra & Bengio 2012 — [JMLR](https://jmlr.org/papers/v13/bergstra12a.html) **[canonical]**.
- RD-Agent(Q) ablation compares bandit vs **random** vs LLM schedulers under a fixed **12-hour wall-clock** budget (44 vs 33 vs 33 loops — i.e., budgets were equal in time, not in evaluations); baselines tuned and reported as **median of 5 seeds** — [arXiv:2505.15155v2](https://arxiv.org/html/2505.15155v2) **[fetched]**.
- FactorMiner includes a "Random Formula Exploration" baseline and applies "the same admission rules to each method and evaluate[s] an equal-sized factor set"; but Alpha101 (Adapted) got up to 10 parameter variants per factor with best chosen — unequal tuning — [arXiv:2602.14670](https://arxiv.org/html/2602.14670v1) **[fetched]**.
- QuantaAlpha compares at "factor-pool level" (~150 factors per LLM method); no random/GP/AlphaGen baseline and no matched evaluation/token budgets; cross-seed IC std 0.0021 over 3 seed combos — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**.

### Inferences — concrete rule set for the fixed backtester and the experiment
1. **Data partition (time-ordered, locked):** `search` window (agent sees metrics) → embargo gap ≥ max(label horizon, feature lookback) bars, rounded up to ≥1 trading day → `validation` window (harness-only, used for selection among agent finalists) → embargo → `test` window that starts **after the LLM's training cutoff** and is evaluated **once** per run, after the run freezes its final candidate(s). The agent API must not expose test-window data, dates, or prices.
2. **Trial ledger (harness-side, append-only):** log every backtest call with config hash, code hash, timestamp, IS metrics, full per-bar return vector. **A trial = any executed evaluation whose metric the searcher can observe**, including failed/discarded candidates, hyperparameter tweaks, re-runs with changed portfolio/risk params, and debugging runs that produce metrics. Identical config hashes are de-duplicated. Unexecuted LLM "ideas" are not trials, but the LLM's prior (published alphas, pretraining on test-period data) is an uncountable trial source → handle via post-cutoff test and originality checks.
3. **Effective N:** report both raw N (ledger count) and N_eff = number of clusters from correlation-based clustering of trial return vectors (ONC, or hierarchical with distance √(0.5(1−ρ))). Use N_raw for the conservative DSR, N_eff as sensitivity.
4. **Accept a strategy only if** (all on data the agent did not select on, or with search-aware deflation): DSR(N_raw) ≥ 0.95 on validation; PBO (CSCV, S=16, Sharpe metric, over all ledger trials of that run) ≤ 0.2 (report value regardless); SPA p ≤ 0.05 vs benchmark (equal-weight universe or zero); HLZ hurdle t(mean IC or mean return) ≥ 3.0; net-of-cost Sharpe > 0 at 2× cost stress.
5. **Minute-bar Sharpe specifics:** compute SR on daily aggregated P&L (minute returns are heavily autocorrelated intraday and non-normal); T = number of days; use skew/kurtosis of daily returns in PSR. For IC-based screening use daily mean of cross-sectional rank IC and Newey–West t-stats.
6. **Agent vs random search fairness protocol:**
   - Same action space: random search samples uniformly (or from a fixed grammar prior) over the **same DSL/config whitelist** the agent uses (D1–D6 subsets).
   - Equal budget in **number of ledger trials** (primary) — also report wall-clock and tokens; RD-Agent-style time budgets favor faster samplers.
   - ≥5 (preferably 10+) independent seeds per arm; report the **distribution** of (a) best-of-N validation score, (b) test score of the candidate each run selects, (c) anytime curves best-so-far vs trials.
   - Selection rule identical for both arms (harness picks top-1/top-k on validation; agent cannot override).
   - Compare arms with a paired/bootstrap test across seeds on the *test-window* metric; apply DSR with each arm's own N.
   - Include a "random search + same admission filters (complexity, correlation cap)" arm so the agent is not credited for filters.
   - Include a leaky-oracle canary (e.g., feature using t+1 return) to verify the harness blocks it — per Gençay, statistics alone will not.

### Gaps
- Could not read full texts of Gençay 2608.27734 or Alpha-R1 2512.23515 (PDF extraction failed); their exact thresholds, PBO values, and trial-counting details are unverified.
- No consensus source on how to count LLM-agent trials when the agent reads metrics but is also influenced by pretraining knowledge; the ledger rule above is an inference.
- No authoritative PBO acceptance threshold exists; 0.2 is a design choice.

---

## Q4. Transaction-cost modeling standards for minute-level backtests (incl. A-share specifics)

### Takeaway
Standard practice: decide on bar t's close, **fill at bar t+1 open or bar t+1 VWAP** (never at the signal bar), charge explicit fees (A-share: commission with 5 CNY min, 5 bps seller stamp duty post-Aug-2023) plus an implicit cost = half-spread + impact, with impact following the **square-root law** `I = Y·σ·√(Q/V)` (Y ≈ O(1)), and cap participation per bar.

### Cited Findings
- Square-root law: average impact of a metaorder of volume Q: `I(Q) = Y·σ·√(Q/φ)`, Y an O(1) coefficient, φ the market's traded volume per unit time; one formulation with daily σ and daily volume cites Y ≈ 2.9 (context unverified); impact exponent empirically anchored near δ = 1/2 — [Bouchaud, "The Square-Root Law of Market Impact" (Substack)](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact), [Maitrier & Bouchaud arXiv:2506.07711](https://arxiv.org/pdf/2506.07711), [Tóth et al., Phys. Rev. X 2011](https://doaj.org/article/aa335a8de8954ff386b6bb094d176225), [Donier et al. arXiv:1412.0141](https://arxiv.org/pdf/1412.0141) **[search snippets]**.
- Qlib default-style fills: `deal_price="close"` example, open_cost 5 bps, close_cost 15 bps, min_cost 5 — [Qlib docs](https://qlib.readthedocs.io/en/latest/component/strategy.html) **[fetched]**; QuantaAlpha uses next-day **open** execution with the same costs — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**.
- FactorMiner (10-minute bars) target is next-bar open-to-close return (implicitly next-bar-open fill), gross of costs, 1–11 bps stress only — [arXiv:2602.14670](https://arxiv.org/html/2602.14670v1) **[fetched]**.
- Stamp duty 0.05% seller-only since 2023-08-28 — [gov.cn](https://english.www.gov.cn/policies/policywatch/202308/28/content_WS64ec5513c6d0868f4e8dee23.html) **[fetched snippet]**.
- Implementation choices (rebalance timing, fill assumptions) are themselves a quantified source of backtest error — "Implementation Risk in Portfolio Backtesting" [arXiv:2603.20319](https://arxiv.org/pdf/2603.20319) **[search snippet only]**.
- Almgren–Chriss temporary + permanent impact framework — [Almgren & Chriss](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) **[canonical]**.

### Inferences — recommended fixed cost model (harness-owned)
- **Timing:** signal uses data ≤ close of bar t; order fills at bar t+1 **VWAP** (default) or open; never same-bar close. Skip lunch break and do not carry signals across the 11:30/13:00 gap without re-evaluation; treat open/close call auctions separately (or exclude first/last bars from trading by default).
- **Explicit fees (A-share):** commission c_comm·notional with max(·, 5 CNY) per order (c_comm configurable, e.g. 2.5 bps); stamp duty 5 bps on sells (10 bps before 2023-08-28 — date-aware); transfer fee ~0.1 bp both sides (verify).
- **Implicit:** `slip_bps = 0.5·spread_bps + Y·σ_bar·√(q / V_bar)·1e4`, with Y ∈ [0.5, 1] as base and stress at 2×; or a simpler fixed floor (e.g. ≥ 5 bps one-way for CSI1000-type names) — take max(model, floor).
- **Participation cap:** fill ≤ 10% (base) of bar volume; remainder unfilled. Limit-up/down: no buys at limit-up, no sells at limit-down; suspended → no fill; T+1 for sells.
- **Report** gross, base-cost and 2×-cost results; acceptance uses base-cost and must stay positive at 2×.

### Gaps
- No primary empirical estimate of Y for A-share minute data was found; the 2.9 figure is unverified.
- No authoritative source found for typical A-share bid-ask spreads by cap tier at minute frequency.

---

## Q5. How existing agent frameworks handle validation vs test, and critiques (overfitting, LLM leakage)

### Takeaway
All four frameworks use a single chronological train/valid/test split, but **none fully specifies that final selection uses validation only**, none applies DSR/PBO/multiple-testing correction, and test periods often **overlap the LLM's pretraining window** (RD-Agent(Q) main test 2017–2020; QuantaAlpha test 2022–2025 with GPT-5.2-class models; AlphaAgent 2021–2024). Independent 2025–2026 work shows memorization inflates LLM backtests and that honest, search-aware evaluation can reject all LLM-discovered strategies.

### Cited Findings
- **RD-Agent(Q):** CSI 300; train 2008–2014, valid 2015–2016, test 2017-01-01 → 2020-08-01; extension CSI 500 / NASDAQ 100 with train 2008–2021, valid 2022–2023, test 2024 → mid-2025, where LLM cutoffs are "completely or nearly prior to the test period". SOTA set updated when a result beats current SOTA; the paper does not explicitly state that this comparison uses only the validation split; Table 1 reports test metrics. Leakage mitigation: LLM sees only schema-level info, not raw data or explicit splits. No discussion of multiple testing found in visible text — [arXiv:2505.15155v2](https://arxiv.org/html/2505.15155v2) **[fetched]**.
- **QuantaAlpha:** train 2016–2020, valid 2021, test 2022-01-01 → 2025-12-26; "the 2022–2025 test window is never accessed during evolution or pool construction"; pool admission validation-gated with |ρ|<0.7, pool capped at 50% of mined factors; zero-shot transfer to CSI 500 and S&P 500; IC t-test over 966 test days (t=7.93) presented as a data-snooping check; **does not discuss LLM pretraining leakage**; internally inconsistent statements on optimal iteration count (decay after ~15 iterations vs best at 11–12) — [arXiv:2602.07085](https://arxiv.org/html/2602.07085) **[fetched]**. Reported IC 0.1501 with GPT-5.2 — [arXiv abs](https://arxiv.org/abs/2602.07085) **[search snippet]**.
- **FactorMiner:** train 2024, test 2025, **no validation split stated**; Top-40 frozen on CSI500 2024 and evaluated 2025 across markets; no multiple-testing correction or leakage checks; gross of costs; uses Gemini 3.0 Flash — [arXiv:2602.14670](https://arxiv.org/html/2602.14670v1) **[fetched]**.
- **AlphaAgent:** evaluation Jan 2021–Dec 2024 on CSI 500/S&P 500 — [arXiv:2502.16789](https://arxiv.org/html/2502.16789v2) **[fetched snippet]** (I did not verify its split details).
- **LLM look-ahead / memorization:**
  - Statistical test for lookahead bias (accuracy rising with a memorization proxy, "LAP"); reported that ~37% of an apparent stock-return predictive effect is amplified by memorization — [arXiv:2512.23847](https://arxiv.org/pdf/2512.23847v1) **[secondary summary; verify]**.
  - FinCAD: inference-time context-aware decoding to suppress outcome memory; cuts in-sample backtest returns by up to 67.1% on memorised dates while leaving 2025 OOS Sharpe within 0.10 of baseline — [arXiv:2605.24564](https://arxiv.org/abs/2605.24564) **[search snippet; preprint]**.
  - MemGuard-Alpha: limits of membership inference for detecting memorization-contaminated LLM signals — [arXiv:2603.26797](https://arxiv.org/pdf/2603.26797) **[search snippet]**.
  - CFA UK community discussion of look-ahead bias in LLM research — [CFA UK](https://connect.cfauk.org/discussion/llms-research-lookahead-bias-for-prediction-tasks) **[search result, low weight]**.
- **Search-aware critique:** Gençay (2026): with leakage-safe tools + trial ledger + DSR/PBO, every LLM-discovered strategy was rejected (2 frontier models, ≤100 candidates, 5 runs) — [arXiv:2608.27734](https://arxiv.org/abs/2608.27734) **[fetched abstract]**.

### Inferences
- For formula-factor agents, leakage is subtler than "LLM knows prices": the LLM knows **which published factors worked** (Alpha101/158, academic anomalies) through dates after the train window. Mitigations: test window after model cutoff; originality vs alpha zoo; report performance of the agent's factors *orthogonalized* to known libraries (Alpha158, Alpha101) on test.
- A fixed, agent-proof backtester should (i) own data access/splits/fills/costs, (ii) be the only path to metrics (so the ledger is complete), (iii) reveal validation metrics only in aggregate (optional: noise/quantization of returned metrics to reduce adaptive overfitting of the validation set), (iv) evaluate test once per frozen run.
- The published frameworks' own comparisons (wall-clock budgets, pool-size matching, unequal tuning) would not meet the fairness protocol in Q3; the planned experiment can contribute exactly this missing control.

### Gaps
- Could not confirm FINSABER's findings (LLM alpha vanishing under survivorship/look-ahead controls) — no primary URL retrieved.
- Could not read RD-Agent(Q) Appendix C.4 (cost numbers, portfolio parameters).
- No public critique specifically targeting RD-Agent/AlphaAgent/QuantaAlpha/FactorMiner results was found beyond the generic LLM-leakage and search-aware-evaluation literature.
