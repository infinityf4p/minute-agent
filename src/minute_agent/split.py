"""Train / validation / locked-test split and the guard that keeps the locked
test period unreadable.

The locked test period (2024-01-01 .. 2026-10-31) can only be read when BOTH:
  1. the environment variable MINUTE_AGENT_UNLOCK_TEST equals UNLOCK_PHRASE, and
  2. the caller passes allow_locked=True to the reading function.

Every data reader in minute_agent.data.store calls `check_range` before it
opens any file, and skips partitions that fall in the locked period. This is
the only switch; there is no config key that can open it.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Period:
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


TRAIN = Period(date(2016, 1, 1), date(2021, 12, 31))
VALID = Period(date(2022, 1, 1), date(2023, 12, 31))
LOCKED_TEST = Period(date(2024, 1, 1), date(2026, 10, 31))

SPLITS = {"train": TRAIN, "valid": VALID, "test": LOCKED_TEST}

UNLOCK_ENV = "MINUTE_AGENT_UNLOCK_TEST"
UNLOCK_PHRASE = "I-UNDERSTAND-THIS-IS-THE-ONE-FINAL-TEST"

# Last date that may be read while locked. Anything after it is blocked too,
# so data newer than the test period can't be used as a back door.
LAST_OPEN_DATE = date(2023, 12, 31)


class LockedDataError(PermissionError):
    """Raised when code tries to read the locked test period."""


_FINAL_EVAL = False   # set only inside final_test_unlock()


def unlocked(allow_locked: bool) -> bool:
    return (allow_locked or _FINAL_EVAL) and os.environ.get(UNLOCK_ENV) == UNLOCK_PHRASE


@contextmanager
def final_test_unlock(reason: str):
    """The one sanctioned way to read the locked period for the final evaluation (M5).
    Needs the env var as well; every opening is appended to runs/locked_test_openings.jsonl."""
    global _FINAL_EVAL
    if os.environ.get(UNLOCK_ENV) != UNLOCK_PHRASE:
        raise LockedDataError(f"set {UNLOCK_ENV} to unlock the test period")
    from minute_agent.paths import runs_root

    log = runs_root() / "locked_test_openings.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"opened_at": datetime.now().isoformat(timespec="seconds"), "reason": reason}) + "\n")
    _FINAL_EVAL = True
    try:
        yield
    finally:
        _FINAL_EVAL = False


def check_range(start: date | None, end: date | None, allow_locked: bool = False) -> None:
    """Raise LockedDataError if [start, end] reaches past LAST_OPEN_DATE while locked."""
    if unlocked(allow_locked):
        return
    if end is None:
        raise LockedDataError(
            "Open-ended read (end=None) is not allowed while the test period is locked; "
            f"pass an explicit end <= {LAST_OPEN_DATE}."
        )
    if end > LAST_OPEN_DATE:
        raise LockedDataError(
            f"Requested data up to {end}, but everything after {LAST_OPEN_DATE} is the locked "
            f"test period. Unlocking needs {UNLOCK_ENV} plus allow_locked=True."
        )
    if start is not None and start > end:
        raise ValueError(f"start {start} > end {end}")


def max_readable_date(allow_locked: bool = False) -> date:
    return LOCKED_TEST.end if unlocked(allow_locked) else LAST_OPEN_DATE


def partition_allowed(year: int, month: int | None, allow_locked: bool = False) -> bool:
    """Whether a (year, month) partition may be opened at all."""
    if unlocked(allow_locked):
        return True
    first_day = date(year, month or 1, 1)
    return first_day <= LAST_OPEN_DATE
