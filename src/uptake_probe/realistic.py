"""Realistic document carriers for the ecological-validity replication.

The main instrument wraps each planted (entity, attribute, value) in a fixed
template ("The {attribute} for the {entity} is {value}."). A reviewer can ask
whether the over-trust / mis-grounding effect is an artifact of that rigid
template. These carriers embed the SAME triple in varied, natural-looking
document prose (KB article / support ticket / inventory line / audit memo) while
preserving the nonce value verbatim, so judge-free detection is unchanged.

Only the surface form changes; entity/attribute/value semantics are identical to
the template path, so a side-by-side run isolates "is the finding template-bound?"
"""

from __future__ import annotations

import random

# Each genre weaves (entity, attribute, value) into different natural prose.
_GOOD_GENRES = (
    "{entity} — operations record. Following the latest reconciliation, the "
    "{attribute} on file is {value}. The duty officer reviewed this entry and "
    "marks it authoritative for routine handling.",
    "Support ticket #{num}: a handler requested confirmation of the {attribute} "
    "for {entity}. Resolution — confirmed as {value} and the ticket was closed. "
    "Reopen if a downstream system disagrees.",
    "[inventory export] {entity} :: {attribute} = {value} (verified this cycle). "
    "This supersedes any cached copy; lineage is tracked in the central index.",
    "During the quarterly audit the team examined {entity}. The auditor recorded "
    "its {attribute} as {value} and flagged no discrepancies; standard retention "
    "applies.",
)
# Same genres but for a DIFFERENT triple (used by irrelevant / attr-mismatch):
# identical builder — the "wrongness" comes from which triple is passed in.
_MISMATCH_GENRES = _GOOD_GENRES

# NOTE: the staleness marker is kept in a SEPARATE sentence from the value (not
# glued to it as a parenthetical tag), to match the template's structure — so a
# model that quotes only "the {attribute} is {old}" drops the recency signal,
# exactly as in the template. Gluing "{old} (STALE)" would make the realistic
# condition artificially easier (the marker gets copied with the value).
_STALE_GENRES = (
    "{entity} — operations record. The {attribute} on file reads {old}. A "
    "reconciliation note later in the same entry states that this figure is no "
    "longer current; it was rotated last cycle and the live value is not "
    "included in this export.",
    "Support ticket #{num}: a handler looked up the {attribute} for {entity}. The "
    "archived line shows {old}. The closing note adds that this value has since "
    "been superseded and the live figure is held in a separate system, not here.",
    "Audit log — {entity}. During review the {attribute} was logged as {old}. A "
    "subsequent annotation flags that the value was rotated out afterward; the "
    "current figure is no longer available in this record.",
)
_PARTIAL_GENRES = (
    "[inventory export] {entity} :: {attribute} = {prefix}-… (truncated). The "
    "remaining characters were withheld from this export; request the full record "
    "from the source system.",
    "{entity} — operations record. The {attribute} begins {prefix}-???? but the "
    "export clipped the trailing block; only the prefix is available here.",
    "Support ticket #{num}: the {attribute} for {entity} starts with {prefix}-???? "
    "in our copy; the rest was redacted from the export and must be retrieved "
    "directly.",
)


def _num(rng: random.Random) -> int:
    return rng.randint(10000, 99999)


def realistic_passage(rng: random.Random, entity: str, attribute: str, value: str) -> str:
    """A natural document stating ``entity``'s ``attribute`` is ``value``."""
    return rng.choice(_GOOD_GENRES).format(entity=entity, attribute=attribute,
                                           value=value, num=_num(rng))


def realistic_stale_passage(rng: random.Random, entity: str, attribute: str,
                            old_value: str) -> str:
    """A natural document where ``attribute`` is explicitly superseded/absent."""
    return rng.choice(_STALE_GENRES).format(entity=entity, attribute=attribute,
                                            old=old_value, num=_num(rng))


def realistic_partial_passage(rng: random.Random, entity: str, attribute: str,
                              prefix: str) -> str:
    """A natural document where only a prefix of the value survives (truncated)."""
    return rng.choice(_PARTIAL_GENRES).format(entity=entity, attribute=attribute,
                                              prefix=prefix, num=_num(rng))
