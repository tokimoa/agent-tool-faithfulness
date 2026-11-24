"""Uptake detection: did a planted-nonce value surface in the answer / trace?

Because values are nonce, exact (separator-tolerant) substring matching is a
clean, judge-free signal. A 5-gram overlap fallback (reused in spirit from the
"copy EXACTLY" detector in the dialogue-compression paper) is provided for the
trace channel where the model may quote the carrier sentence rather than the
bare value.
"""

from __future__ import annotations

import re

from .schema import Instance, NonceFact
from .util import normalize, surface_forms


def detect_value(value: str, text: str) -> bool:
    """True if any accepted surface form of ``value`` appears in ``text``."""
    norm = normalize(text)
    return any(form in norm for form in surface_forms(value))


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(reference: str, text: str, n: int = 5) -> float:
    """Fraction of ``reference`` n-grams that appear verbatim in ``text``."""
    ref_tokens = normalize(reference).split()
    txt_tokens = normalize(text).split()
    ref_ngrams = _ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    txt_ngrams = _ngrams(txt_tokens, n)
    return len(ref_ngrams & txt_ngrams) / len(ref_ngrams)


def fact_in_text(fact: NonceFact, text: str) -> bool:
    """Uptake of a fact: the nonce VALUE is present (separator-tolerant).

    We deliberately do NOT accept carrier-sentence n-gram overlap as a fallback.
    A model that echoes the question phrasing ("The X for the Y is ...") WITHOUT
    the value (e.g. "... is not provided in the search results") would otherwise
    false-positive. The nonce value is the only ground truth for uptake. The
    ``ngram_overlap`` helper is kept for diagnostics, not for the uptake decision.
    """
    return detect_value(fact.value, text)


def detect_uptake(instance: Instance, answer: str, trace: str = "") -> dict[str, dict[str, bool]]:
    """Per necessary fact: was it taken up in the answer / in the trace?

    Keyed by entity. The trace channel lets us separate "ignored entirely" from
    "read but dropped from the answer" (the Neglect signal).
    """
    out: dict[str, dict[str, bool]] = {}
    for fact in instance.necessary_facts:
        out[fact.entity] = {
            "answer": fact_in_text(fact, answer),
            "trace": bool(trace) and fact_in_text(fact, trace),
        }
    return out


_ABSTAIN = re.compile(
    r"\b(not found|no record|no result|no match|cannot (be )?determine|"
    r"could ?n.?t (be )?(found|determined|retrieved|located)|unavailable|"
    r"unable to|not (\w+ly )?(available|provided|present|listed|included|recorded|"
    r"current|stated|specified|mentioned|in the (search )?results)|"
    r"no (matching|valid) (record|result|data|entry)|insufficient|n/?a|unknown|"
    r"do(es)? not (contain|provide|include|have|list|match|specify|mention)|"
    r"only (mention|list|provide|contain|show|include|specif)\w*|"
    r"is (about|for) (a |the )?(different|another)|not (about|for) (the|this)|"
    r"mismatch|wrong (entity|record|result)|returned no|"
    # recency recognition (STALE quality): flagging a value as outdated is NOT a
    # confabulation — the agent has recognised the asked (current) value is absent.
    r"superseded|no longer (current|valid|in use|available|included|listed|recorded)|"
    r"rotated out|"
    r"(previous|former|prior|outdated|stale|old) (key|value|code|token|passphrase|"
    r"reference|tag|id|entry|record)|"
    r"redacted|access (restricted|denied)|error)\b",
    re.IGNORECASE,
)
_VALUE_PAT = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")


def detect_abstention(text: str) -> bool:
    """True if the answer signals the value is unavailable / cannot be given."""
    return bool(_ABSTAIN.search(text))


def has_value_code(text: str) -> bool:
    """True if the answer presents a nonce value-code (a fabricated/borrowed value).

    Under an unobtainable (bad-result) condition the true value was never shown,
    so any code in the answer is a confabulation or a value lifted from an
    irrelevant result.
    """
    return bool(_VALUE_PAT.search(text))


def confab_source(instance: Instance, answer: str) -> str | None:
    """Mechanism of a confabulation on an IRRELEVANT (unobtainable) instance:
    - "decoy" : the answer copies the irrelevant result's value (mis-grounding —
      the agent used the wrong-but-retrieved value).
    - "novel" : a value-code not present anywhere (pure fabrication).
    - None    : no value-code in the answer (not a confabulation).
    """
    if not has_value_code(answer):
        return None
    decoys = [r.wrong_value for r in instance.results if r.is_necessary and r.wrong_value]
    if any(detect_value(d, answer) for d in decoys):
        return "decoy"
    return "novel"


def followed_contradiction(instance: Instance, answer: str) -> dict[str, bool]:
    """For each contradictory distractor: did the answer adopt its WRONG value?"""
    out: dict[str, bool] = {}
    for r in instance.contradictory_results:
        if r.wrong_value is not None:
            out[r.contradicts_entity or r.result_id] = detect_value(r.wrong_value, answer)
    return out
