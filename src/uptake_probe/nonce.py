"""Deterministic generation of planted-nonce facts.

The whole instrument rests on the values being *un-memorizable*: random
alphanumeric codes that cannot appear in pretraining, so that finding the value
in the answer is causal evidence the agent read THIS result (not its prior).
"""

from __future__ import annotations

import random

from .schema import NonceFact

_CONS = "bcdfghjklmnpqrstvwxz"
_VOW = "aeiou"
_LET = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O to avoid digit confusion
_DIG = "23456789"                  # no 0/1 to avoid letter confusion

_ATTRIBUTES = (
    "activation code",
    "access key",
    "clearance ID",
    "dispatch token",
    "ledger reference",
    "calibration code",
    "vault passphrase",
    "routing tag",
)

# Pronounceable but invented stems, so entities read as plausible proper nouns
# while remaining novel. Combined with a numeric suffix for uniqueness.
_STEMS = (
    "Zephur", "Quattle", "Brindor", "Vexmar", "Onculi", "Drennoth",
    "Yulvane", "Pacthrem", "Klovist", "Marneth", "Surabek", "Tindovar",
)
_KINDS = ("protocol", "registry", "module", "consortium", "directive", "array", "ledger", "syndicate")


def nonce_value(rng: random.Random) -> str:
    """Return a code like ``QX94-MP07`` (2 letters, 2 digits, '-', repeat)."""
    def block() -> str:
        return (
            rng.choice(_LET) + rng.choice(_LET) + rng.choice(_DIG) + rng.choice(_DIG)
        )

    return f"{block()}-{block()}"


def nonce_entity(rng: random.Random) -> str:
    stem = rng.choice(_STEMS)
    num = rng.randint(1000, 9999)
    kind = rng.choice(_KINDS)
    return f"{stem}-{num} {kind}"


def make_nonce_fact(rng: random.Random, attribute: str | None = None) -> NonceFact:
    return NonceFact(
        entity=nonce_entity(rng),
        attribute=attribute if attribute is not None else rng.choice(_ATTRIBUTES),
        value=nonce_value(rng),
    )


def near_miss_entity(rng: random.Random, entity: str) -> str:
    """A near-identical entity: same stem + kind, different number.

    "Zephur-4417 protocol" -> "Zephur-5912 protocol". Makes a maximally plausible
    mismatch (an agent that doesn't check the exact id is easily fooled).
    """
    head, _, kind = entity.partition(" ")
    stem = head.rsplit("-", 1)[0]
    return f"{stem}-{rng.randint(1000, 9999)} {kind}"


def different_value(rng: random.Random, avoid: str) -> str:
    """A nonce value distinct from ``avoid`` (for contradictory distractors)."""
    for _ in range(50):
        v = nonce_value(rng)
        if v != avoid:
            return v
    raise RuntimeError("could not draw a distinct nonce value")  # pragma: no cover


def different_attribute(rng: random.Random, avoid: str) -> str:
    """An attribute distinct from ``avoid`` (for attribute-mismatch results: the
    result is about the RIGHT entity but a DIFFERENT attribute than asked)."""
    choices = [a for a in _ATTRIBUTES if a != avoid]
    return rng.choice(choices)
