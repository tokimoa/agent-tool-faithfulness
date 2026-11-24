"""Scoring: per-fact OND outcome, plus cell-level aggregation.

OND precedence for a necessary fact NOT used in the answer:
  Distraction (a wrong value was followed) > Omission (never observed) > Neglect.
In passive delivery every result is in-context, so Omission cannot occur there —
that asymmetry is exactly the agentic-loop cost we want to surface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .detect import (
    detect_abstention,
    detect_uptake,
    followed_contradiction,
    has_value_code,
)
from .scaffolds import RunRecord
from .schema import OND, Instance


@dataclass
class FactOutcome:
    entity: str
    used: bool
    ond: OND
    in_trace: bool
    contra_present: bool
    contra_followed: bool


def evaluate_run(instance: Instance, record: RunRecord) -> list[FactOutcome]:
    uptake = detect_uptake(instance, record.answer, record.trace)
    contra = followed_contradiction(instance, record.answer)
    contra_entities = {r.contradicts_entity for r in instance.contradictory_results}
    observed = record.observed_necessary

    outcomes: list[FactOutcome] = []
    for fact in instance.necessary_facts:
        e = fact.entity
        used = uptake[e]["answer"]
        in_trace = uptake[e]["trace"]
        c_present = e in contra_entities
        c_followed = contra.get(e, False)
        if used:
            ond = OND.USED
        elif c_followed:
            ond = OND.DISTRACTION
        elif e not in observed:
            ond = OND.OMISSION
        else:
            ond = OND.NEGLECT
        outcomes.append(FactOutcome(e, used, ond, in_trace, c_present, c_followed))
    return outcomes


def classify_error_awareness(instance: Instance, record: RunRecord) -> str:
    """For an UNOBTAINABLE (bad-result) instance, did the agent abstain, recover,
    confabulate, or stay vague?

    - confabulate : presents a value-code (fabricated, or lifted from an
      irrelevant/distractor result) — "satisfied with a bad result".
    - recover     : abstained AND re-searched (tried another query).
    - abstain     : signalled the value is unavailable, no fabricated value.
    - vague       : neither a value nor a clear abstention.
    """
    ans = record.answer.strip()
    if not ans:
        return "no_answer"   # gave up without fabricating (often ran out of steps)
    # Abstention takes PRECEDENCE over has_value_code: a model that says "the
    # result does not contain X; it only mentions Y (CODE)" is RECOGNISING the
    # mismatch (good), even though it quotes the irrelevant value. Only treat a
    # value as confabulation when the answer asserts it WITHOUT flagging a problem.
    if detect_abstention(ans):
        return "recover" if record.searches > 1 else "abstain"
    if has_value_code(ans):
        return "confabulate"   # presented a value as the answer, no recognition
    return "vague"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(per_instance: list[list[FactOutcome]]) -> dict:
    """Aggregate a cell (one mode x K) into headline metrics."""
    flat = [o for inst in per_instance for o in inst]
    if not flat:
        return {"n_facts": 0}

    uptake = _mean([1.0 if o.used else 0.0 for o in flat])
    coverage = _mean([1.0 if all(o.used for o in inst) else 0.0 for inst in per_instance])

    failures = [o for o in flat if o.ond is not OND.USED]
    ond = Counter(o.ond.value for o in failures)
    ond_share = {k: ond.get(k, 0) / len(flat) for k in ("omission", "neglect", "distraction")}

    # Contradiction resolution: for a necessary fact that has a contradictory
    # distractor, classify the answer by which value(s) appear. This separates
    # "committed to the lie" (wrong only) from "hedged" (both present) — the raw
    # "wrong value present" rate conflates them.
    contra_cells = [o for o in flat if o.contra_present]
    cfr = None
    contradiction = None
    if contra_cells:
        cfr = _mean([1.0 if o.contra_followed else 0.0 for o in contra_cells])
        contradiction = {
            "right_only": round(_mean([1.0 if (o.used and not o.contra_followed) else 0.0
                                       for o in contra_cells]), 4),
            "wrong_only": round(_mean([1.0 if ((not o.used) and o.contra_followed) else 0.0
                                       for o in contra_cells]), 4),
            "both": round(_mean([1.0 if (o.used and o.contra_followed) else 0.0
                                 for o in contra_cells]), 4),
            "neither": round(_mean([1.0 if ((not o.used) and not o.contra_followed) else 0.0
                                    for o in contra_cells]), 4),
        }

    return {
        "n_facts": len(flat),
        "n_instances": len(per_instance),
        "uptake_rate": round(uptake, 4),
        "coverage": round(coverage, 4),
        "ond_share": {k: round(v, 4) for k, v in ond_share.items()},
        "contradiction_following_rate": round(cfr, 4) if cfr is not None else None,
        "contradiction": contradiction,
    }


def consistency(used_by_trial: list[list[bool]]) -> float:
    """pass^k-style stability: fraction of cells where all trials agree on `used`.

    ``used_by_trial[cell]`` is the list of per-trial `used` bits for one
    (instance, fact). Only informative for stochastic backends (trials>1).
    """
    if not used_by_trial:
        return 1.0
    agree = sum(1 for bits in used_by_trial if len(set(bits)) <= 1)
    return round(agree / len(used_by_trial), 4)
