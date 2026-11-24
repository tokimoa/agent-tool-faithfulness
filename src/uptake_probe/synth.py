"""Synthetic instance generator: planted facts x position x count x distractor type.

Template-based carrier passages keep the pilot deterministic and CPU-only; the
design doc (§3.2) calls for MLX paraphrase to add naturalness later. For the
pilot we only need clean ground truth and controllable difficulty.
"""

from __future__ import annotations

import random

from .domains import get_domain
from .nonce import different_value, nonce_value
from .schema import (
    SOURCE_AUTHORITATIVE,
    SOURCE_UNVERIFIED,
    DistractorType,
    Instance,
    NonceFact,
    ResultQuality,
    ToolResult,
)

# Filler sentences wrapped around the carrier fact so a result is a short passage
# rather than a bare sentence (more realistic; also dilutes trivial copying).
_FILLER = (
    "This entry was last reconciled during the quarterly audit.",
    "Handlers should confirm scope before acting on this record.",
    "The record is retained per the standard retention window.",
    "Cross-references are maintained in the central index.",
)


def _carrier_passage(rng: random.Random, sentence: str) -> str:
    pre, post = rng.sample(_FILLER, 2)
    return f"{pre} {sentence} {post}"


# Optional lexical signal that an IRRELEVANT result is a *mismatch*, layered onto
# the (otherwise silent) wrong-entity passage at increasing explicitness. If
# confabulation falls only once the mismatch is spelled out in WORDS — not when
# the entity id merely differs — the agent is reacting to the string, not auditing
# entity identity. A model that stays high even at "hard" ignores an explicit
# textual warning (the deepest grounding failure). Counterpart to ``verify_entity``
# in ``_question``: there the cue is in the QUESTION, here it is in the RESULT.
_MISMATCH_CUE = {
    "none": "",
    "soft": "This record may concern a related but distinct item. ",
    "hard": "No exact match was found for the requested entity; showing the "
            "closest available record. ",
}


def _center_positions(k: int, n: int) -> list[int]:
    """Indices clustered around the middle of ``k`` slots (pilot uses center)."""
    mid = (k - 1) / 2.0
    # pick the n slots closest to the middle, then sort by index
    order = sorted(range(k), key=lambda i: (abs(i - mid), i))
    return sorted(order[:n])


def _positions(k: int, n: int, position: str) -> list[int]:
    if position == "center":
        return _center_positions(k, n)
    if position == "early":
        return list(range(n))
    if position == "late":
        return list(range(k - n, k))
    if position == "spread":
        if n == 1:
            return [k // 2]
        step = max(1, (k - 1) // (n - 1))
        return sorted({min(k - 1, i * step) for i in range(n)})
    raise ValueError(f"unknown position: {position!r}")


def _necessary_result(rng: random.Random, fact: NonceFact, idx: int,
                      quality: ResultQuality = ResultQuality.GOOD,
                      mismatch_signal: str = "none", style: str = "template",
                      dom=None, attr_label_mode: str = "wrong") -> ToolResult:
    """A necessary result. When ``quality`` != GOOD the value is UNOBTAINABLE
    from it, so the correct behaviour is abstain/recover, not confabulate.

    ``mismatch_signal`` ("none"/"soft"/"hard") prepends an explicit mismatch cue
    to an IRRELEVANT result (no effect on other qualities); see ``_MISMATCH_CUE``.
    ``style`` ("template"/"realistic") swaps the fixed carrier template for varied
    natural document prose. ``dom`` is the active Domain (entity/attribute/genres);
    the value stays a nonce so judge-free scoring is unchanged across domains.
    """
    if dom is None:
        dom = get_domain("access")
    real = style == "realistic"
    rec = style == "record"   # multi-field structured record (richer realistic form)

    def triple(t: NonceFact) -> str:
        return (dom.realistic_good(rng, t.entity, t.attribute, t.value) if real
                else _carrier_passage(rng, t.carrier_sentence))

    decoy = None
    if quality is ResultQuality.GOOD:
        text = dom.record_good(rng, fact.entity, fact.attribute, fact.value) if rec else triple(fact)
    elif quality is ResultQuality.ERROR:
        text = "ERROR 404: no matching record was found in the registry for this query."
    elif quality is ResultQuality.EMPTY:
        text = "(the record returned no content)"
    elif quality is ResultQuality.REDACTED:
        text = _carrier_passage(
            rng, f"The {fact.attribute} for the {fact.entity} is [REDACTED — access restricted].")
    elif quality in (ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_NEAR,
                     ResultQuality.IRRELEVANT_FAR):
        # a passage about a DIFFERENT entity, at a controlled "plausibility" distance.
        # Its value (the decoy) is recorded so we can tell mis-grounding (copying
        # the decoy) from pure fabrication (a novel value).
        if quality is ResultQuality.IRRELEVANT_NEAR:        # same attr, near-identical entity
            other = NonceFact(dom.near_miss(rng, fact.entity), fact.attribute, nonce_value(rng))
        elif quality is ResultQuality.IRRELEVANT_FAR:       # different attr AND entity
            other = dom.new_fact(rng)
        else:                                               # mid: same attr, different entity
            other = NonceFact(dom.entity(rng), fact.attribute, nonce_value(rng))
        body = (dom.record_irrelevant(rng, other.entity, other.attribute, other.value)
                if rec else triple(other))
        text = _MISMATCH_CUE[mismatch_signal] + body
        decoy = other.value
    elif quality is ResultQuality.ATTR_MISMATCH:
        # RIGHT entity, WRONG attribute: an entity-identity check passes here, so
        # this dissociates "skipped entity-check" from "use any on-topic value".
        other = NonceFact(fact.entity, dom.diff_attr(rng, fact.attribute), nonce_value(rng))
        text = (dom.record_attr_mismatch(rng, fact.entity, fact.attribute,
                                         other.attribute, other.value,
                                         label_mode=attr_label_mode)
                if rec else triple(other))
        decoy = other.value
    elif quality is ResultQuality.STALE:
        # RIGHT entity+attribute, but the value is explicitly superseded and the
        # current one is absent: needs a recency check, not an identity check.
        old = nonce_value(rng)
        if rec:
            text = dom.record_stale(rng, fact.entity, fact.attribute, old)
        elif real:
            text = dom.realistic_stale(rng, fact.entity, fact.attribute, old)
        else:
            text = _carrier_passage(
                rng, f"As of an earlier cycle (now superseded), the {fact.attribute} for "
                f"the {fact.entity} was {old}. It has since been rotated and the "
                "current value is not included in this record.")
        decoy = old
    elif quality is ResultQuality.PARTIAL:
        # RIGHT entity+attribute, but the value is truncated: the full value is
        # unobtainable, so any COMPLETE value-code in the answer is a fabricated
        # completion (expected NOVEL, contrasting the decoy-copy of the mismatches).
        prefix = fact.value.split("-")[0]
        if rec:
            text = dom.record_partial(rng, fact.entity, fact.attribute, prefix)
        elif real:
            text = dom.realistic_partial(rng, fact.entity, fact.attribute, prefix)
        else:
            text = _carrier_passage(
                rng, f"The {fact.attribute} for the {fact.entity} begins with {prefix}-"
                "???? (the remaining characters were truncated and not returned).")
        # no decoy: there is no complete wrong value to copy
    else:  # pragma: no cover
        raise ValueError(f"unknown quality: {quality!r}")
    return ToolResult(
        result_id=f"r{idx}",
        full_text=text,
        snippet=f"{fact.entity} — {fact.attribute} on record",  # always looks available
        source=SOURCE_AUTHORITATIVE,
        is_necessary=True,
        fact=fact,
        quality=quality,
        wrong_value=decoy,  # the decoy value present in the irrelevant result (or None)
    )


def _distractor_result(
    rng: random.Random,
    idx: int,
    dtype: DistractorType,
    necessary_facts: list[NonceFact],
    style: str = "template",
    dom=None,
) -> ToolResult:
    if dom is None:
        dom = get_domain("access")

    def carrier(t: NonceFact) -> str:
        if style == "record":
            return dom.record_good(rng, t.entity, t.attribute, t.value)
        return (dom.realistic_good(rng, t.entity, t.attribute, t.value)
                if style == "realistic" else _carrier_passage(rng, t.carrier_sentence))

    if dtype is DistractorType.IRRELEVANT:
        f = dom.new_fact(rng)
        return ToolResult(
            result_id=f"r{idx}",
            full_text=carrier(f),
            snippet=f"{f.entity} — {f.attribute} on record",
            source=SOURCE_UNVERIFIED,
            is_necessary=False,
            distractor_type=dtype,
        )
    if dtype is DistractorType.HARD_NEGATIVE:
        # same attribute as a necessary fact, different entity (tempting confusion)
        target = rng.choice(necessary_facts)
        f = NonceFact(entity=dom.entity(rng), attribute=target.attribute, value=nonce_value(rng))
        return ToolResult(
            result_id=f"r{idx}",
            full_text=carrier(f),
            snippet=f"{f.entity} — {f.attribute} on record",
            source=SOURCE_UNVERIFIED,
            is_necessary=False,
            distractor_type=dtype,
        )
    if dtype is DistractorType.CONTRADICTORY:
        target = rng.choice(necessary_facts)
        wrong = different_value(rng, target.value)
        liar = NonceFact(entity=target.entity, attribute=target.attribute, value=wrong)
        return ToolResult(
            result_id=f"r{idx}",
            full_text=carrier(liar),
            snippet=f"{target.entity} — {target.attribute} (unverified)",
            source=SOURCE_UNVERIFIED,
            is_necessary=False,
            distractor_type=dtype,
            contradicts_entity=target.entity,
            wrong_value=wrong,
        )
    raise ValueError(f"unknown distractor type: {dtype!r}")  # pragma: no cover


def _question(facts: list[NonceFact], authority_cue: bool = True,
              verify_entity: bool = False) -> str:
    if authority_cue:
        # spoon-feeds the disambiguation rule (easy: tests instruction-following)
        instruction = (
            "Using only the official registry results, answer precisely. "
            "Report each requested value exactly as written."
        )
    else:
        # no rule given: the agent must infer authority from the source labels
        # (official_registry vs unverified_note) — the harder RQ4 condition.
        instruction = (
            "Answer precisely using the retrieved results. "
            "Report each requested value exactly as written."
        )
    if verify_entity:
        instruction += (
            " IMPORTANT: before reporting any value, verify the result is about the "
            "EXACT entity named; if a result is about a different entity, treat the "
            "value as not available."
        )
    asks = " ".join(f"({i + 1}) What is the {f.attribute} for the {f.entity}?"
                    for i, f in enumerate(facts))
    return f"{instruction} {asks}"


def make_instance(
    seed: int,
    k: int,
    n_necessary: int = 1,
    position: str = "center",
    distractor_types: tuple[DistractorType, ...] = (DistractorType.IRRELEVANT,),
    include_contradictory: bool = False,
    authority_cue: bool = True,
    necessary_quality: ResultQuality = ResultQuality.GOOD,
    verify_entity: bool = False,
    mismatch_signal: str = "none",
    style: str = "template",
    domain: str = "access",
    attr_label_mode: str = "wrong",
) -> Instance:
    """Build one evidence puzzle.

    ``k`` total results = ``n_necessary`` necessary + the rest distractors. When
    ``include_contradictory`` is set, one distractor slot is forced to be a
    contradictory result that lies about a necessary fact (for RQ4).
    ``authority_cue`` controls whether the question explicitly tells the agent to
    trust the official registry (easy) or leaves it to infer authority (hard).
    """
    if n_necessary > k:
        raise ValueError("n_necessary cannot exceed k")
    rng = random.Random(seed)
    dom = get_domain(domain)

    facts = [dom.new_fact(rng) for _ in range(n_necessary)]
    pos = _positions(k, n_necessary, position)

    # Decide the distractor type for each non-necessary slot.
    n_distractors = k - n_necessary
    dtypes: list[DistractorType] = []
    if include_contradictory and n_distractors > 0:
        dtypes.append(DistractorType.CONTRADICTORY)
    while len(dtypes) < n_distractors:
        dtypes.append(rng.choice(distractor_types))
    rng.shuffle(dtypes)

    # Lay out results: necessary facts at ``pos``, distractors elsewhere.
    results: list[ToolResult | None] = [None] * k
    for slot, fact in zip(pos, facts):
        results[slot] = _necessary_result(rng, fact, slot, quality=necessary_quality,
                                          mismatch_signal=mismatch_signal, style=style,
                                          dom=dom, attr_label_mode=attr_label_mode)
    unobtainable = ({f.entity for f in facts}
                    if necessary_quality is not ResultQuality.GOOD else set())
    di = 0
    for slot in range(k):
        if results[slot] is None:
            results[slot] = _distractor_result(rng, slot, dtypes[di], facts, style=style,
                                               dom=dom)
            di += 1

    final = [r for r in results if r is not None]
    return Instance(
        instance_id=f"s{seed}_k{k}_n{n_necessary}_{position}"
        + ("_contra" if include_contradictory else "")
        + ("" if authority_cue else "_nocue")
        + ("" if necessary_quality is ResultQuality.GOOD else f"_{necessary_quality.value}")
        + ("" if mismatch_signal == "none" else f"_sig-{mismatch_signal}")
        + ("" if style == "template" else f"_{style}")
        + ("" if domain == "access" else f"_{domain}"),
        question=_question(facts, authority_cue=authority_cue, verify_entity=verify_entity),
        necessary_facts=facts,
        results=final,
        necessary_positions=pos,
        k=k,
        n_necessary=n_necessary,
        position=position,
        seed=seed,
        unobtainable_entities=unobtainable,
        meta={
            "include_contradictory": include_contradictory,
            "authority_cue": authority_cue,
            "necessary_quality": necessary_quality.value,
            "mismatch_signal": mismatch_signal,
            "style": style,
            "domain": domain,
            "distractor_types": [d.value for d in dtypes],
        },
    )
