"""Hash lock for the evaluator source.

FROZEN.sha256 lists the sha256 of every .py file in this package. The runner
refuses to run, and the test suite fails, if the files don't match. After an
intentional, human-reviewed change, re-freeze with:

    python -m minute_agent.evaluator.integrity --freeze
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
MANIFEST = PKG / "FROZEN.sha256"


class EvaluatorModifiedError(RuntimeError):
    pass


def _files() -> list[Path]:
    return sorted(p for p in PKG.glob("*.py"))


def _hash(p: Path) -> str:
    # normalise line endings so the hash is the same on Windows and macOS checkouts
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current() -> dict[str, str]:
    return {p.name: _hash(p) for p in _files()}


def combined_hash(h: dict[str, str] | None = None) -> str:
    h = h or current()
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(h.items())).encode()).hexdigest()[:16]


def freeze() -> str:
    h = current()
    MANIFEST.write_text("".join(f"{v}  {k}\n" for k, v in sorted(h.items())), encoding="utf-8", newline="\n")
    return combined_hash(h)


def verify() -> str:
    if not MANIFEST.exists():
        raise EvaluatorModifiedError("evaluator/FROZEN.sha256 is missing")
    want = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v, k = line.split()
            want[k] = v
    have = current()
    if want != have:
        changed = sorted(set(want.items()) ^ set(have.items()))
        raise EvaluatorModifiedError(f"evaluator source differs from FROZEN.sha256: {changed}")
    return combined_hash(have)


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        print("frozen:", freeze())
    else:
        print("ok:", verify())
