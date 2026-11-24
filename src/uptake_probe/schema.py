"""Core data types for the Tool-Result Uptake benchmark.

An ``Instance`` is one "agentic evidence puzzle": a question that can only be
answered by using the planted-nonce value(s) carried in one or more *necessary*
tool results, possibly hidden among *distractor* results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DistractorType(str, Enum):
    IRRELEVANT = "irrelevant"          # unrelated entity/attribute
    HARD_NEGATIVE = "hard_negative"    # same attribute, different (tempting) entity
    CONTRADICTORY = "contradictory"    # asserts a WRONG value for a necessary fact


class DeliveryMode(str, Enum):
    PASSIVE = "passive"            # all results delivered in one prompt (lost-in-the-middle foil)
    AGENTIC = "agentic"           # results arrive over tool-call turns (the agent must fetch them)
    PASSIVE_LONG = "passive_long"  # passive, padded to agentic length (de-confounds length vs loop)
    AGENTIC_COMMIT = "agentic_commit"  # action-consequential: the agent must WRITE the value into a
    #                                    downstream system (COMMIT) or hand off (ESCALATE) — measures
    #                                    acting on a mis-grounded value, not merely reporting it


class OND(str, Enum):
    """Failure decomposition for a necessary fact that was NOT used in the answer."""

    USED = "used"               # success (not a failure)
    OMISSION = "omission"       # the carrying result was never observed/fetched
    NEGLECT = "neglect"         # observed (or in-context) but absent from the answer
    DISTRACTION = "distraction"  # a wrong/irrelevant value was used instead


class ResultQuality(str, Enum):
    """Quality of a (necessary) result — for the error-awareness axis.

    When not GOOD, the asked fact is UNOBTAINABLE from that result, so the
    correct behaviour is to abstain/recover, not to confabulate a value.
    """

    GOOD = "good"             # carries the correct value
    ERROR = "error"           # tool returned an error ("record not found")
    EMPTY = "empty"           # tool returned nothing
    IRRELEVANT = "irrelevant"  # DIFFERENT entity, SAME attribute (mid plausibility)
    IRRELEVANT_NEAR = "irrelevant_near"  # near-identical entity (same stem), same attr — most plausible
    IRRELEVANT_FAR = "irrelevant_far"    # different entity AND attribute — least plausible
    REDACTED = "redacted"     # value present but redacted/restricted
    # --- failure taxonomy (B): mismatches an entity-identity check would NOT catch ---
    ATTR_MISMATCH = "attr_mismatch"  # SAME entity, DIFFERENT attribute — needs an attribute check
    STALE = "stale"           # SAME entity+attribute, value explicitly superseded — needs a recency check
    PARTIAL = "partial"       # SAME entity+attribute, value truncated — needs a completeness check


# Authoritative vs non-authoritative source labels. Used so an inter-tool-result
# conflict has a well-defined "right" answer (the authoritative necessary result)
# and a "wrong" one (the contradictory distractor).
SOURCE_AUTHORITATIVE = "official_registry"
SOURCE_UNVERIFIED = "unverified_note"


@dataclass(frozen=True)
class NonceFact:
    """A planted, un-memorizable (entity, attribute, value) triple."""

    entity: str       # e.g. "Zephur-4417 protocol"
    attribute: str    # e.g. "activation code"
    value: str        # the nonce to be used, e.g. "QX94-MP07"

    @property
    def question(self) -> str:
        return f"What is the {self.attribute} for the {self.entity}?"

    @property
    def carrier_sentence(self) -> str:
        return f"The {self.attribute} for the {self.entity} is {self.value}."


@dataclass
class ToolResult:
    """One returned result the agent may read."""

    result_id: str
    full_text: str            # the carrier passage (what FETCH returns / passive shows)
    snippet: str              # short descriptor shown by SEARCH
    source: str               # SOURCE_AUTHORITATIVE / SOURCE_UNVERIFIED
    is_necessary: bool
    fact: NonceFact | None = None          # set when is_necessary
    distractor_type: DistractorType | None = None
    contradicts_entity: str | None = None  # for CONTRADICTORY: the entity it lies about
    wrong_value: str | None = None         # for CONTRADICTORY: the wrong value asserted
    quality: "ResultQuality" = None        # for necessary results (defaults to GOOD in __post_init__)

    def __post_init__(self) -> None:
        if self.quality is None:
            self.quality = ResultQuality.GOOD


@dataclass
class Instance:
    instance_id: str
    question: str
    necessary_facts: list[NonceFact]
    results: list[ToolResult]       # ordered; index == delivery/context position
    necessary_positions: list[int]  # indices in ``results`` of the necessary facts
    k: int
    n_necessary: int
    position: str                   # "early" | "center" | "late" | "spread"
    seed: int
    unobtainable_entities: set[str] = field(default_factory=set)  # error-awareness axis
    meta: dict = field(default_factory=dict)

    @property
    def gold_values(self) -> list[str]:
        return [f.value for f in self.necessary_facts]

    @property
    def contradictory_results(self) -> list[ToolResult]:
        return [r for r in self.results if r.distractor_type is DistractorType.CONTRADICTORY]
