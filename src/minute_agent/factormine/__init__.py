"""Minute-factor mining with a factor library (separate from the pipeline search in minute_agent.agent).

A factor has two stages:
  1. intraday: minute bars of one stock-day (bars up to the decision bar only) -> a few numbers,
     written either in the minute DSL (dsl.py) or as sandboxed Python code;
  2. combine: an expression over the daily panel (intraday outputs plus lagged daily variables)
     with time-series and cross-sectional operators.
The evaluator (evaluate.py) alone decides whether a factor enters the library (library.py).
See docs/factormine/design.md.
"""
