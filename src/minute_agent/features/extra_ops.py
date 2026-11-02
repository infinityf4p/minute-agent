"""Operators for M3: the bucketed intraday sequence for sequence models, and
agent-written custom features (v2 action space).

Both are ordinary registered operators, so they go through the same feature
cache and the same look-ahead check as everything else.
"""

from __future__ import annotations

import ast
import hashlib
import math

import numpy as np
import polars as pl

from minute_agent.features.ops import KEYS, register

# ---------------------------------------------------------------------------
# sequence features

SEQ_PREFIX = "sq_"
SEQ_CHANNELS = ["ret", "range", "vshare", "vwapdev", "cumret", "upfrac"]


def seq_shape(d: int, bucket_min: int) -> int:
    """Number of time steps: the opening auction plus ceil(d / bucket_min) buckets."""
    return 1 + math.ceil(d / bucket_min)


@register("seq_bars")
def op_seq_bars(m, daily, d, bucket_min=10):
    """Bars up to the decision time aggregated into buckets of `bucket_min` minutes.
    Step 0 is the opening auction, its return is measured against pre_close.
    Output columns sq_<step>_<channel>, float32, roughly unit scale."""
    b = int(bucket_min)
    T = seq_shape(d, b)
    pre = daily.select("symbol", "date", "pre_close")
    x = m.join(pre, on=KEYS, how="left").with_columns(
        pl.when(pl.col("bidx") == 0).then(0).otherwise((pl.col("bidx") - 1) // b + 1).cast(pl.Int32).alias("step")
    )
    g = x.group_by(KEYS + ["step"]).agg(
        pl.col("close").last().alias("c"),
        pl.col("high").max().alias("h"),
        pl.col("low").min().alias("l"),
        pl.col("volume").sum().alias("v"),
        pl.col("amount").sum().alias("a"),
        (pl.col("lr") > 0).mean().alias("up"),
        pl.col("pre_close").first().alias("pc"),
        pl.col("open").first().alias("o"),
    )
    grid = g.select(KEYS).unique().join(pl.DataFrame({"step": list(range(T))}, schema={"step": pl.Int32}), how="cross")
    g = grid.join(g, on=KEYS + ["step"], how="left").sort(KEYS + ["step"])
    g = g.with_columns(
        pl.col("c").forward_fill().over(KEYS),
        pl.col("pc").forward_fill().backward_fill().over(KEYS),
        pl.col("v").fill_null(0.0), pl.col("a").fill_null(0.0),
    ).with_columns(
        pl.col("c").shift(1).over(KEYS).alias("c_prev"),
        pl.col("c").first().over(KEYS).alias("c0"),
        pl.col("v").sum().over(KEYS).alias("vtot"),
    )
    feats = g.select(
        *KEYS, "step",
        (pl.when(pl.col("step") == 0).then((pl.col("c") / pl.col("pc")).log())
         .otherwise((pl.col("c") / pl.col("c_prev")).log()) * 100).clip(-10, 10).alias("ret"),
        ((pl.col("h") / pl.col("l")).log() * 100).fill_null(0.0).clip(0, 10).alias("range"),
        (pl.when(pl.col("vtot") > 0).then(pl.col("v") / pl.col("vtot") * T).otherwise(0.0)).clip(0, 20).alias("vshare"),
        (pl.when(pl.col("v") > 0).then((pl.col("c") / (pl.col("a") / pl.col("v")) - 1) * 100).otherwise(0.0)).clip(-5, 5).alias("vwapdev"),
        ((pl.col("c") / pl.col("c0")).log() * 100).clip(-20, 20).alias("cumret"),
        (pl.col("up").fill_null(0.5) - 0.5).alias("upfrac"),
    ).with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32) for c in SEQ_CHANNELS])
    wide = feats.pivot(on="step", index=KEYS, values=SEQ_CHANNELS, aggregate_function="first")
    # rename to sq_<step>_<channel> with zero-padded step, columns ordered step-major
    ren = {}
    for col in wide.columns:
        if col in KEYS:
            continue
        ch, step = col.rsplit("_", 1)
        ren[col] = f"{SEQ_PREFIX}{int(step):02d}_{ch}"
    wide = wide.rename(ren)
    ordered = [f"{SEQ_PREFIX}{t:02d}_{c}" for t in range(T) for c in SEQ_CHANNELS]
    return wide.select(KEYS + ordered)


def seq_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith(SEQ_PREFIX)]


# ---------------------------------------------------------------------------
# custom features (agent-written code)

ALLOWED_IMPORTS = {"polars", "numpy", "math"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr",
    "delattr", "vars", "input", "breakpoint", "help", "exit", "quit", "memoryview", "dir", "type",
    "object", "super", "classmethod", "staticmethod", "property",
}
FORBIDDEN_ATTR_PREFIX = ("read_", "scan_", "write_", "sink_", "from_", "load", "save", "to_csv", "to_parquet")
FORBIDDEN_ATTRS = {"fromfile", "tofile", "loadtxt", "savetxt", "genfromtxt", "memmap", "ctypeslib", "DataSource",
                   "Config", "plugins", "sql", "SQLContext", "io", "os", "sys", "map_elements", "map_batches",
                   "map_groups", "apply", "pipe", "collect_async", "random"}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
                           "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
                           "ValueError", "Exception", "print"]}


class CustomFeatureError(ValueError):
    pass


def static_check(code: str) -> None:
    """Reject code that could do IO, reach the network, or escape the sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CustomFeatureError(f"syntax error: {e}") from e
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feature"]
    if not fn or len(fn[0].args.args) != 3:
        raise CustomFeatureError("code must define `def feature(m, daily, d):`")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            if any(mm not in ALLOWED_IMPORTS for mm in mods):
                raise CustomFeatureError(f"import not allowed: {mods}; allowed: {sorted(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CustomFeatureError(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_") or a in FORBIDDEN_ATTRS or a.startswith(FORBIDDEN_ATTR_PREFIX):
                raise CustomFeatureError(f"attribute not allowed: .{a}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Lambda)):
            raise CustomFeatureError(f"construct not allowed: {type(node).__name__}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and ("__" in node.value):
            raise CustomFeatureError("string constants containing '__' are not allowed")


_COMPILED: dict[str, object] = {}


def compile_feature(code: str):
    h = hashlib.sha1(code.encode()).hexdigest()
    if h not in _COMPILED:
        static_check(code)
        ns = {"__builtins__": SAFE_BUILTINS, "pl": pl, "np": np, "math": math}
        safe_imports = {"polars": pl, "numpy": np, "math": math}

        def _imp(name, *a, **k):
            if name.split(".")[0] not in safe_imports:
                raise ImportError(name)
            return safe_imports[name.split(".")[0]]
        ns["__builtins__"] = {**SAFE_BUILTINS, "__import__": _imp}
        exec(compile(code, "<custom_feature>", "exec"), ns)
        _COMPILED[h] = ns["feature"]
    return _COMPILED[h]


@register("custom")
def op_custom(m, daily, d, name="custom", code=""):
    """Agent-written feature. Output columns are prefixed with the feature name."""
    fn = compile_feature(code)
    out = fn(m, daily, d)
    if not isinstance(out, pl.DataFrame) or any(k not in out.columns for k in KEYS):
        raise CustomFeatureError("feature() must return a polars DataFrame with symbol and date columns")
    vals = [c for c in out.columns if c not in KEYS]
    if not vals or len(vals) > 10:
        raise CustomFeatureError("feature() must return 1..10 value columns")
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:24]
    out = out.select(KEYS + [pl.col(c).cast(pl.Float64, strict=False).alias(f"cf_{safe}_{c}"[:60]) for c in vals])
    if out.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise CustomFeatureError("feature() returned duplicate (symbol, date) rows")
    return m.select(KEYS).unique().join(out, on=KEYS, how="left")
