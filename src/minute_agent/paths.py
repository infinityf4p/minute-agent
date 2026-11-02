"""Filesystem layout. Everything under data/, cache/ and runs/ is git-ignored."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root of standardized parquet datasets. Override with MINUTE_AGENT_DATA."""
    return Path(os.environ.get("MINUTE_AGENT_DATA", REPO_ROOT / "data"))


def cache_root() -> Path:
    """Root of the feature cache. Override with MINUTE_AGENT_CACHE."""
    return Path(os.environ.get("MINUTE_AGENT_CACHE", REPO_ROOT / "cache"))


def runs_root() -> Path:
    """Root of experiment outputs and the SQLite ledger. Override with MINUTE_AGENT_RUNS."""
    return Path(os.environ.get("MINUTE_AGENT_RUNS", REPO_ROOT / "runs"))


def dataset_dir(dataset: str) -> Path:
    """A dataset is a folder like data/sim_cn or data/cn."""
    return data_root() / dataset
