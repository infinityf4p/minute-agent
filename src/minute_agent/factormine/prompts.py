"""Prompts and the mechanism catalogue for the LLM arms."""

from __future__ import annotations

from minute_agent.factormine import dsl
from minute_agent.factormine.config import THRESHOLDS
from minute_agent.factormine.data import DAILY_VAR_HELP

MECHANISMS = {
    "tail_flow": "late-session trading activity and price moves (who trades near the close, and how)",
    "open_impact": "the opening auction and the first half hour: gaps, early volume, early reversals",
    "intraday_reversal": "price moves within the day that tend to reverse the next day",
    "intraday_momentum": "intraday trends or persistent buying that continue the next day",
    "pv_divergence": "relations between price changes and volume (confirmation or divergence)",
    "vol_asymmetry": "upside versus downside volatility or volume within the day",
    "return_shape": "shape of the intraday return distribution (skew, tails, concentration)",
    "jumps": "large one-minute moves: their number, direction, timing and aftermath",
    "volume_concentration": "how concentrated volume is over the day; bursts and quiet periods",
    "vwap_position": "where the price sits relative to the day's VWAP and how it got there",
    "liquidity_impact": "price impact per unit traded (Amihud-like), spreads proxied by bar ranges",
    "informed_trading": "returns during high-volume minutes versus low-volume minutes (informed vs noise trading)",
    "am_pm_split": "differences between morning and afternoon behaviour",
    "multi_day_dynamics": "how an intraday characteristic changes or accumulates over several days",
    "overnight_vs_intraday": "gaps between yesterday's close and today's open versus today's intraday path",
    "attention_shocks": "abnormal activity relative to the stock's own recent days (volume, range, jumps)",
}


def system_prompt(arm: str, k: int, seeds: list[dict], code_help: str) -> str:
    th = THRESHOLDS
    if arm == "llm_code":
        intraday = ("INTRADAY STAGE: write Python code (the \"code\" field) defining `def feature(m, daily, d):`.\n"
                    + code_help)
        shape = ('{"name": str, "hypothesis": str, "code": "<python source>", "combine": <daily DSL expression '
                 'using the output column names of your code>}')
    else:
        intraday = ("INTRADAY STAGE: 1 to 3 named nodes in the minute DSL (the \"intraday\" field, "
                    'e.g. {"a": {...}, "b": {...}}).\n' + dsl.doc())
        shape = ('{"name": str, "hypothesis": str, "intraday": {"a": node, ...}, "combine": <daily DSL expression '
                 'using a, b, ... and daily variables>}')
    seed_lines = "\n".join(f"  - {s['id']}: {s['desc']}" for s in seeds)
    return f"""You are a quantitative researcher mining intraday alpha factors from minute bars of China A-shares.

TASK
Every trading day at 14:50 (decision bar d = 230) a factor gives one number per stock, using only data known at
14:50: today's minute bars up to 14:50 and earlier days. It should predict the stock's return from the VWAP of
14:51-15:00 today to the same window on the next trading day, relative to the cross-sectional mean.
Universe: a fixed list of about 600 A-share stocks of all sizes.

HOW FACTORS ARE JUDGED (by an automatic evaluator; you never see future data)
Mining period 2020-2021, two folds (2020, 2021). For each day the factor's cross-sectional rank is regressed on
style exposures (size, today's return to 14:50, 1/5/20-day returns, 20-day volatility, 20-day turnover); the
residual's rank correlation with the label is the residual IC. A factor enters the library only if
  - coverage >= {th['min_coverage']} of stock-days, |t| of the mean daily residual IC >= {th['min_res_t']},
    and the residual IC has the same sign in 2020 and in 2021;
  - its absolute cross-sectional rank correlation with every library member is < {th['max_corr']}.
    A factor correlated with exactly one non-fixed member replaces it if its |residual IC| is
    {th['replace_ratio']}x larger.
The library starts with these fixed, well-known intraday factors; new factors must be different from them:
{seed_lines}
Simply restating today's return, recent returns, volatility, turnover or size gains nothing: they are
neutralized. Feedback shows raw_ic (before neutralization) and res_ic: a large raw_ic with a small res_ic means the
factor mostly restates a style; change what it measures, not its smoothing.
The goal is many distinct, robust factors, not one strong one.

A FACTOR HAS TWO STAGES
{intraday}

{dsl.combine_doc(DAILY_VAR_HELP)}

OUTPUT
Call propose_factors with exactly {k} factors, each {shape}.
Keep names short (snake_case). The hypothesis says in one or two sentences why the factor should predict returns.
Keep your private reasoning brief (a few hundred words) and then call the tool.
"""


CODE_HELP = """- m: minute bars of all stocks for one month, already cut at the decision bar (bidx <= d), columns
  symbol, date, ts, bidx (0 = 09:30 auction, 1..120 = 09:31..11:30, 121..240 = 13:01..15:00), open, high, low,
  close, volume (shares), amount (CNY), lr (1-minute log return within the day, null on bar 0).
  Sorted by symbol, date, bidx.
- daily: daily bars (symbol, date, open, high, low, close, pre_close, volume, amount, turnover, adj_factor,
  up_limit, down_limit, is_suspended, is_st, float_mv, total_mv) from about 70 days before the month to its end.
  Use only rows of EARLIER dates (e.g. shift(1).over("symbol")), except pre_close, adj_factor, up_limit,
  down_limit, is_st, is_suspended of the same date (known before the open). Prices are unadjusted.
- Return a polars DataFrame with columns symbol, date and 1..3 numeric columns (snake_case names, not the names of
  daily variables), one row per stock-day. These names are the variables of the combine stage.
  Exactly symbol, date and at most 3 more columns: after joins, finish with
  .select("symbol", "date", "<col1>", ...) so helper columns (pre_close, means, counts) are dropped.
- polars has no pl.sign/pl.abs/pl.log functions: use expression methods (pl.col("x").sign(), .abs(), .log1p()).
- Only `import polars as pl`, `import numpy as np`, `import math`. No IO, no dunder attributes, no
  eval/exec/getattr, no map_elements/apply, no lambda, no classes. Use vectorized polars (group_by/agg, over).
- Runtime limit: 60 seconds per month (about 600 stocks x 21 days x 231 bars). A look-ahead check runs first:
  values for day T must not change when data after 14:50 of day T is removed.
Example:
import polars as pl
def feature(m, daily, d):
    late = pl.col("bidx") > d - 20
    return m.group_by("symbol", "date").agg(
        (pl.col("volume").filter(late).sum() / pl.col("volume").sum()).alias("late_vol_share"),
        pl.col("lr").filter(late).sum().alias("late_ret"))"""


def tool_schema(arm: str, k: int) -> dict:
    item = {"type": "object", "properties": {
        "name": {"type": "string"}, "hypothesis": {"type": "string"},
        "combine": {"type": "object", "description": "daily DSL expression"}}, "required": ["name", "hypothesis", "combine"]}
    if arm == "llm_code":
        item["properties"]["code"] = {"type": "string", "description": "python source defining feature(m, daily, d)"}
        item["required"].append("code")
    else:
        item["properties"]["intraday"] = {"type": "object", "description": "1..3 named minute DSL nodes"}
        item["required"].append("intraday")
    return {"type": "function", "function": {
        "name": "propose_factors", "description": f"Submit exactly {k} candidate factors for evaluation.",
        "parameters": {"type": "object", "properties": {"factors": {"type": "array", "items": item}},
                       "required": ["factors"]}}}
