"""Small deterministic helpers shared across modules.

Determinism matters: every nonce, placement, and simulated decision must be
reproducible from a seed so experiments can be replayed exactly (CLAUDE.md:
全実験パラメータ・シード値を記録).
"""

from __future__ import annotations

import hashlib
import re


def stable_hash(*parts: object) -> int:
    """A process-independent 63-bit hash of the given parts.

    Python's built-in ``hash`` is salted per process, which would break
    reproducibility. We hash the repr of the parts with blake2b instead.
    """
    payload = "␟".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    return _WS.sub(" ", text.lower()).strip()


def surface_forms(value: str) -> tuple[str, ...]:
    """Accepted written variants of a nonce value (with/without separators).

    A model may drop the hyphen ("QX94-MP07" -> "QX94MP07") or space it; we
    accept those so detection measures *uptake*, not formatting compliance.
    """
    v = normalize(value)
    forms = {v, v.replace("-", ""), v.replace("-", " ")}
    return tuple(sorted(forms, key=len, reverse=True))
