"""Two small languages for factors.

Minute DSL (intraday stage). One node turns the bars of one stock-day, up to the decision bar d, into one number:
    {"agg": <agg>, "x": <field>, "y": <field, corr only>, "win": <window>, "cond": <condition or null>}

Daily DSL (combine stage). An expression tree over the daily panel:
    {"var": <intraday output name or daily variable>}
    {"op": <unary>, "x": node}            {"op": <binary>, "x": node, "y": node}
    {"op": <ts op>, "x": node, "n": int}  {"op": "ts_corr", "x": node, "y": node, "n": int}
Time-series operators run over each stock's own trading days (rows), backwards only.
"""

from __future__ import annotations

import hashlib
import json
import random

import polars as pl

FIELDS = {
    "ret": "1-minute log return (null on the auction bar)",
    "absret": "absolute 1-minute log return",
    "volume": "shares traded in the bar",
    "amount": "value traded in the bar (CNY)",
    "vwapdev": "close / running VWAP of the day so far - 1",
    "hl": "high / low - 1 of the bar",
    "cumret": "close / auction price (bar 0) - 1",
    "bidx": "bar index (time of day), 0 = 09:30 auction, 1..120 morning, 121..240 afternoon",
}
WINDOWS = {
    "auction": "bar 0 only (opening call auction)",
    "open30": "bars 1..30 (09:31-10:00)",
    "am": "bars 1..120 (morning)",
    "pm": "bars 121..d (afternoon up to the decision bar)",
    "last30": "the 30 bars ending at d",
    "last60": "the 60 bars ending at d",
    "all": "bars 1..d",
}
CONDS = {
    None: "no condition",
    "up": "bars with ret > 0",
    "down": "bars with ret < 0",
    "highvol": "bars with volume above the day's median bar volume (bars 1..d)",
    "lowvol": "bars with volume below the day's median bar volume",
    "bigmove": "bars with absret above 2x the day's mean absret",
}
AGGS = {
    "mean": "mean of x", "std": "std of x", "sum": "sum of x", "skew": "skewness of x", "kurt": "kurtosis of x",
    "max": "max of x", "min": "min of x", "first": "x at the first selected bar", "last": "x at the last selected bar",
    "share": "sum of x over selected bars / sum of x over bars 0..d (x in volume, amount, absret)",
    "corr": "correlation of x and y over selected bars",
    "count_frac": "number of selected bars / number of bars in the window (x ignored)",
    "argmax": "time of the bar with the largest x among selected bars (bidx / 240)",
}
SHARE_FIELDS = ("volume", "amount", "absret")
UNARY = {"neg": "-x", "abs": "|x|", "sign": "sign(x)", "log1p_abs": "sign(x) * log(1+|x|)",
         "sqrt_abs": "sign(x) * sqrt(|x|)", "cs_rank": "cross-sectional rank in [0,1] per day",
         "cs_zscore": "cross-sectional z-score per day"}
BINARY = {"add": "x + y", "sub": "x - y", "mul": "x * y", "div": "x / y (null when |y| tiny)"}
TS = {"ts_mean": "mean over the last n days (including today)", "ts_std": "std over the last n days",
      "ts_sum": "sum over the last n days", "ts_max": "max over n days", "ts_min": "min over n days",
      "ts_zscore": "(x - ts_mean) / ts_std over n days", "ts_minmax": "(x - ts_min) / (ts_max - ts_min) over n days",
      "delta": "x - x n days ago", "ts_corr": "rolling correlation of x and y over n days"}
TS_N = (3, 5, 10, 20, 40, 60)
MAX_NODES = 15
MAX_DEPTH = 6


class DSLError(ValueError):
    pass


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def h(obj) -> str:
    return hashlib.sha1(canon(obj).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- minute DSL

def check_node(n: dict) -> dict:
    if not isinstance(n, dict):
        raise DSLError("an intraday node must be an object")
    extra = set(n) - {"agg", "x", "y", "win", "cond"}
    if extra:
        raise DSLError(f"unknown intraday keys {sorted(extra)}")
    agg, x, y, win, cond = n.get("agg"), n.get("x"), n.get("y"), n.get("win", "all"), n.get("cond")
    if agg not in AGGS:
        raise DSLError(f"agg must be one of {list(AGGS)}, got {agg!r}")
    if agg != "count_frac" and x not in FIELDS:
        raise DSLError(f"x must be one of {list(FIELDS)}, got {x!r}")
    if win not in WINDOWS:
        raise DSLError(f"win must be one of {list(WINDOWS)}, got {win!r}")
    if cond not in CONDS:
        raise DSLError(f"cond must be one of {[c for c in CONDS]}, got {cond!r}")
    if agg == "corr":
        if y not in FIELDS or y == x:
            raise DSLError("corr needs a field y different from x")
    elif y is not None:
        raise DSLError("y is only used by corr")
    if agg == "share" and x not in SHARE_FIELDS:
        raise DSLError(f"share needs x in {SHARE_FIELDS}")
    if win == "auction" and (agg not in ("sum", "share", "last") or x not in ("volume", "amount")):
        raise DSLError("the auction window has one bar: use sum/share/last of volume or amount")
    out = {"agg": agg, "win": win, "cond": cond}
    if agg != "count_frac":
        out["x"] = x
    if agg == "corr":
        out["y"] = y
    return out


def prepare_bars(m: pl.DataFrame, d: int) -> pl.DataFrame:
    """Minute bars (symbol, date, bidx, open, high, low, close, volume, amount) cut at d, plus DSL fields/masks."""
    k = ["symbol", "date"]
    m = m.filter(pl.col("bidx") <= d).sort("symbol", "date", "bidx")
    m = m.with_columns((pl.col("close") / pl.col("close").shift(1)).log().over(k).alias("ret"))
    m = m.with_columns(
        pl.col("ret").abs().alias("absret"),
        (pl.col("close") / (pl.col("amount").cum_sum() / pl.col("volume").cum_sum()).over(k) - 1).alias("vwapdev"),
        (pl.col("high") / pl.col("low") - 1).alias("hl"),
        (pl.col("close") / pl.col("close").first().over(k) - 1).alias("cumret"),
        pl.col("bidx").cast(pl.Float64).alias("bidx_f"),
    )
    regular = pl.col("bidx") >= 1
    med = pl.col("volume").filter(regular).median().over(k)
    mabs = pl.col("absret").filter(regular).mean().over(k)
    return m.with_columns(
        (pl.col("ret") > 0).fill_null(False).alias("c_up"),
        (pl.col("ret") < 0).fill_null(False).alias("c_down"),
        (regular & (pl.col("volume") > med)).alias("c_highvol"),
        (regular & (pl.col("volume") < med)).alias("c_lowvol"),
        (regular & (pl.col("absret") > 2 * mabs)).fill_null(False).alias("c_bigmove"),
    ).with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in ("vwapdev", "hl", "cumret")])


def _win(win: str, d: int) -> pl.Expr:
    b = pl.col("bidx")
    return {"auction": b == 0, "open30": b.is_between(1, 30), "am": b.is_between(1, 120),
            "pm": b.is_between(121, d), "last30": b.is_between(d - 29, d), "last60": b.is_between(d - 59, d),
            "all": b.is_between(1, d)}[win]


def _col(f: str) -> pl.Expr:
    return pl.col("bidx_f") if f == "bidx" else pl.col(f)


def node_expr(n: dict, d: int, alias: str) -> pl.Expr:
    w = _win(n["win"], d)
    sel = w & pl.col(f"c_{n['cond']}") if n["cond"] else w
    agg = n["agg"]
    if agg == "count_frac":
        e = sel.sum() / w.sum()
    else:
        x = _col(n["x"]).filter(sel)
        if agg == "kurt":
            e = x.kurtosis()
        elif agg in ("mean", "std", "sum", "skew", "max", "min", "first", "last"):
            e = getattr(x, agg)()
        elif agg == "share":
            e = x.sum() / _col(n["x"]).sum()
        elif agg == "corr":
            e = pl.corr(x, _col(n["y"]).filter(sel))
        elif agg == "argmax":
            e = pl.col("bidx_f").filter(sel).sort_by(x).last() / 240.0
        else:
            raise DSLError(agg)
    e = e.cast(pl.Float64)
    return pl.when(e.is_finite()).then(e).alias(alias)


def eval_nodes(bars: pl.DataFrame, nodes: dict[str, dict], d: int) -> pl.DataFrame:
    return bars.group_by("symbol", "date").agg([node_expr(n, d, k) for k, n in nodes.items()])


# ----------------------------------------------------------------------------- daily DSL

def check_expr(e, variables: set[str], depth: int = 0, count: list | None = None) -> dict:
    count = count if count is not None else [0]
    count[0] += 1
    if count[0] > MAX_NODES:
        raise DSLError(f"combine expression has more than {MAX_NODES} nodes")
    if depth > MAX_DEPTH:
        raise DSLError(f"combine expression deeper than {MAX_DEPTH}")
    if not isinstance(e, dict):
        raise DSLError("combine nodes must be objects")
    if e.get("op") == "var" and isinstance(e.get("x"), str) and set(e) == {"op", "x"}:
        e = {"var": e["x"]}               # tolerate a common LLM spelling of a variable
    if "var" in e:
        if set(e) != {"var"} or e["var"] not in variables:
            raise DSLError(f"unknown variable {e.get('var')!r}; available: {sorted(variables)}")
        return {"var": e["var"]}
    op = e.get("op")
    if op in UNARY:
        _keys(e, {"op", "x"})
        return {"op": op, "x": check_expr(e["x"], variables, depth + 1, count)}
    if op in BINARY:
        _keys(e, {"op", "x", "y"})
        return {"op": op, "x": check_expr(e["x"], variables, depth + 1, count),
                "y": check_expr(e["y"], variables, depth + 1, count)}
    if op in TS:
        two = op == "ts_corr"
        _keys(e, {"op", "x", "n"} | ({"y"} if two else set()))
        n = e["n"]
        if n not in TS_N or (op == "delta" and n > 20):
            raise DSLError(f"n for {op} must be one of {[k for k in TS_N if op != 'delta' or k <= 20]}")
        if n < 5 and op in ("ts_std", "ts_zscore", "ts_corr"):
            raise DSLError(f"{op} needs n >= 5")
        out = {"op": op, "x": check_expr(e["x"], variables, depth + 1, count), "n": n}
        if two:
            out["y"] = check_expr(e["y"], variables, depth + 1, count)
        return out
    raise DSLError(f"unknown op {op!r}; unary {list(UNARY)}, binary {list(BINARY)}, time-series {list(TS)}")


def _keys(e: dict, need: set) -> None:
    if set(e) != need:
        raise DSLError(f"op {e.get('op')} needs exactly the keys {sorted(need)}, got {sorted(e)}")


def vars_used(e: dict) -> set[str]:
    if "var" in e:
        return {e["var"]}
    return set().union(*(vars_used(e[k]) for k in ("x", "y") if k in e))


def _fin(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_finite()).then(e)


def evaluate(frame: pl.DataFrame, e: dict, out: str = "f") -> pl.DataFrame:
    """Add column `out` = expression e. frame has one row per stock-day, sorted by symbol, date.
    Every node is materialized as its own column, so window expressions are never nested."""
    counter = [0]

    def rec(fr: pl.DataFrame, node: dict) -> tuple[pl.DataFrame, str]:
        if "var" in node:
            return fr, node["var"]
        fr, xc = rec(fr, node["x"])
        yc = None
        if "y" in node:
            fr, yc = rec(fr, node["y"])
        counter[0] += 1
        name = f"__n{counter[0]}"
        ex = _op_expr(node, pl.col(xc), pl.col(yc) if yc else None).cast(pl.Float64)
        return fr.with_columns(pl.when(ex.is_finite()).then(ex).alias(name)), name

    fr, col = rec(frame, e)
    return fr.with_columns(pl.col(col).cast(pl.Float64).alias(out)).drop([c for c in fr.columns if c.startswith("__n")])


def _op_expr(e: dict, x: pl.Expr, y: pl.Expr | None) -> pl.Expr:
    op = e["op"]
    if op == "neg":
        return -x
    if op == "abs":
        return x.abs()
    if op == "sign":
        return x.sign()
    if op == "log1p_abs":
        return x.sign() * x.abs().log1p()
    if op == "sqrt_abs":
        return x.sign() * x.abs().sqrt()
    if op == "cs_rank":
        return (x.rank() / x.count()).over("date")
    if op == "cs_zscore":
        return _fin((x - x.mean().over("date")) / x.std().over("date"))
    if op in BINARY:
        if op == "div":
            return pl.when(y.abs() > 1e-12).then(x / y)
        return {"add": x + y, "sub": x - y, "mul": x * y}[op]
    n = e["n"]
    ms = max(2, n // 2)
    s = pl.col("symbol")
    if op == "ts_mean":
        return x.rolling_mean(n, min_samples=ms).over(s)
    if op == "ts_sum":
        return x.rolling_sum(n, min_samples=ms).over(s)
    if op == "ts_std":
        return x.rolling_std(n, min_samples=ms).over(s)
    if op == "ts_max":
        return x.rolling_max(n, min_samples=ms).over(s)
    if op == "ts_min":
        return x.rolling_min(n, min_samples=ms).over(s)
    if op == "ts_zscore":
        return _fin((x - x.rolling_mean(n, min_samples=ms)) / x.rolling_std(n, min_samples=ms)).over(s)
    if op == "ts_minmax":
        lo, hi = x.rolling_min(n, min_samples=ms), x.rolling_max(n, min_samples=ms)
        return _fin((x - lo) / (hi - lo)).over(s)
    if op == "delta":
        return (x - x.shift(n)).over(s)
    if op == "ts_corr":
        return _fin(pl.rolling_corr(x, y, window_size=n, min_samples=ms)).over(s)
    raise DSLError(op)


# ----------------------------------------------------------------------------- random generator

def random_node(rng: random.Random) -> dict:
    while True:
        agg = rng.choice(list(AGGS))
        win = rng.choice(list(WINDOWS))
        cond = rng.choice(list(CONDS))
        n = {"agg": agg, "win": win, "cond": cond}
        if agg != "count_frac":
            n["x"] = rng.choice(SHARE_FIELDS) if agg == "share" else rng.choice(list(FIELDS))
        if agg == "corr":
            n["y"] = rng.choice([f for f in FIELDS if f != n["x"]])
        if win == "auction":
            n = {"agg": rng.choice(["sum", "share", "last"]), "x": rng.choice(["volume", "amount"]),
                 "win": "auction", "cond": None}
        if agg == "count_frac" and cond is None:
            continue
        try:
            return check_node(n)
        except DSLError:
            continue


def random_expr(rng: random.Random, intraday_vars: list[str], daily_vars: list[str], depth: int = 0) -> dict:
    p_leaf = 0.15 + 0.3 * depth
    if depth >= 3 or rng.random() < p_leaf:
        pool = intraday_vars if rng.random() < 0.8 else daily_vars
        return {"var": rng.choice(pool)}
    kind = rng.choices(["unary", "binary", "ts"], weights=[2, 3, 4])[0]
    if kind == "unary":
        return {"op": rng.choice(list(UNARY)), "x": random_expr(rng, intraday_vars, daily_vars, depth + 1)}
    if kind == "binary":
        return {"op": rng.choice(list(BINARY)), "x": random_expr(rng, intraday_vars, daily_vars, depth + 1),
                "y": random_expr(rng, intraday_vars, daily_vars, depth + 1)}
    op = rng.choice(list(TS))
    n = rng.choice([k for k in TS_N if (op != "delta" or k <= 20) and (k >= 5 or op not in ("ts_std", "ts_zscore", "ts_corr"))])
    out = {"op": op, "x": random_expr(rng, intraday_vars, daily_vars, depth + 1), "n": n}
    if op == "ts_corr":
        out["y"] = random_expr(rng, intraday_vars, daily_vars, depth + 1)
    return out


def random_factor(rng: random.Random, daily_vars: list[str]) -> dict:
    k = rng.choice([1, 1, 2])
    intraday = {name: random_node(rng) for name in ["a", "b"][:k]}
    while True:
        e = random_expr(rng, list(intraday), daily_vars)
        try:
            e = check_expr(e, set(intraday) | set(daily_vars))
        except DSLError:
            continue
        if vars_used(e) & set(intraday):
            break
    return {"name": f"rnd_{h([intraday, e])[:8]}", "mechanism": "random", "hypothesis": "",
            "intraday": intraday, "combine": e}


def doc() -> str:
    lines = ["MINUTE DSL (intraday stage, one number per stock-day from bars up to the decision bar d):",
             '  {"agg": <agg>, "x": <field>, "y": <field, corr only>, "win": <window>, "cond": <condition or null>}',
             "  fields: " + "; ".join(f"{k} = {v}" for k, v in FIELDS.items()),
             "  windows: " + "; ".join(f"{k} = {v}" for k, v in WINDOWS.items()),
             "  conditions: " + "; ".join(f"{k} = {v}" for k, v in CONDS.items()),
             "  aggs: " + "; ".join(f"{k} = {v}" for k, v in AGGS.items()),
             "  The auction window allows only sum/share/last of volume or amount."]
    return "\n".join(lines)


def combine_doc(daily_help: dict[str, str]) -> str:
    return "\n".join([
        "DAILY DSL (combine stage, an expression tree evaluated per stock-day on the daily panel):",
        '  {"var": name} | {"op": unary, "x": e} | {"op": binary, "x": e, "y": e} | {"op": ts, "x": e, "n": int}'
        ' | {"op": "ts_corr", "x": e, "y": e, "n": int}',
        "  unary: " + "; ".join(f"{k} = {v}" for k, v in UNARY.items()),
        "  binary: " + "; ".join(f"{k} = {v}" for k, v in BINARY.items()),
        "  time-series (per stock, backwards over its trading days): " + "; ".join(f"{k} = {v}" for k, v in TS.items()),
        f"  n in {list(TS_N)} (delta: n <= 20; ts_std/ts_zscore/ts_corr: n >= 5). At most {MAX_NODES} nodes, depth {MAX_DEPTH}.",
        "  variables: your intraday outputs (by name) and these daily variables known at the decision time: "
        + "; ".join(f"{k} = {v}" for k, v in daily_help.items()),
        '  A variable is written {"var": "a"} (never {"op": "var", ...}). Example: '
        '{"op": "ts_mean", "x": {"op": "div", "x": {"var": "a"}, "y": {"var": "b"}}, "n": 10}.',
        "  The factor's sign does not matter: the evaluator orients it by its mining-period IC.",
    ])
