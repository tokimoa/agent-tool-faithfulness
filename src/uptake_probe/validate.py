"""Deterministic structural validators for generated benchmark instances.

The bulk of data-quality control is mechanical and must NOT rely on an LLM:
- the planted nonce value appears verbatim exactly where it should,
- the decoy (wrong-but-retrieved value) appears exactly once in a mismatch result,
- an attribute-mismatch result does NOT leak the asked attribute,
- a partial result leaks NO complete value-code,
- staleness / truncation markers are present where the condition requires them.

These run for FREE on every instance and gate it before any (optional) LLM
naturalness check. Keeping scoring-adjacent checks deterministic preserves the
benchmark's judge-free guarantee; the LLM is reserved for the one subjective
axis (does realistic prose read naturally), handled separately in qc.py.
"""

from __future__ import annotations

from .detect import detect_abstention, detect_value, has_value_code
from .schema import Instance, ResultQuality


def structural_issues(inst: Instance) -> list[str]:
    """Return a list of structural problems; empty list == the instance is clean."""
    issues: list[str] = []
    nec = [r for r in inst.results if r.is_necessary]
    if len(nec) != inst.n_necessary:
        issues.append(f"necessary count {len(nec)} != n_necessary {inst.n_necessary}")

    for r in nec:
        fact = r.fact
        q = r.quality
        text = r.full_text
        if q is ResultQuality.GOOD:
            # the true value must be present verbatim and be the ONLY code
            if not detect_value(fact.value, text):
                issues.append(f"{r.result_id}: GOOD missing true value {fact.value}")
        elif q in (ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_NEAR,
                   ResultQuality.IRRELEVANT_FAR, ResultQuality.ATTR_MISMATCH):
            # the decoy must be present; the TRUE value must be absent (unobtainable)
            if r.wrong_value is None:
                issues.append(f"{r.result_id}: {q.value} has no decoy recorded")
            elif not detect_value(r.wrong_value, text):
                issues.append(f"{r.result_id}: {q.value} decoy {r.wrong_value} not in text")
            if detect_value(fact.value, text):
                issues.append(f"{r.result_id}: {q.value} leaks the TRUE value {fact.value}")
            if q is ResultQuality.ATTR_MISMATCH and fact.attribute in text:
                issues.append(f"{r.result_id}: ATTR_MISMATCH leaks asked attribute "
                              f"'{fact.attribute}'")
            if q in (ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_NEAR,
                     ResultQuality.IRRELEVANT_FAR) and fact.entity in text:
                issues.append(f"{r.result_id}: {q.value} leaks the asked entity")
        elif q is ResultQuality.STALE:
            if r.wrong_value is None or not detect_value(r.wrong_value, text):
                issues.append(f"{r.result_id}: STALE missing old value in text")
            if not detect_abstention(text):
                issues.append(f"{r.result_id}: STALE lacks a recency/superseded marker")
            if detect_value(fact.value, text):
                issues.append(f"{r.result_id}: STALE leaks the current value")
        elif q is ResultQuality.PARTIAL:
            if r.wrong_value is not None:
                issues.append(f"{r.result_id}: PARTIAL should have no decoy")
            if has_value_code(text):
                issues.append(f"{r.result_id}: PARTIAL leaks a complete value-code")
            if detect_value(fact.value, text):
                issues.append(f"{r.result_id}: PARTIAL leaks the full true value")
        elif q in (ResultQuality.ERROR, ResultQuality.EMPTY, ResultQuality.REDACTED):
            if has_value_code(text) and q is not ResultQuality.REDACTED:
                issues.append(f"{r.result_id}: {q.value} unexpectedly contains a value-code")

    # the gold answer key must be recoverable and unique per necessary fact
    if len(inst.gold_values) != len(set(inst.gold_values)):
        issues.append("duplicate gold values across necessary facts")
    return issues


def is_clean(inst: Instance) -> bool:
    return not structural_issues(inst)
