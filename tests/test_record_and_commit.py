"""Tests for the two validity upgrades:

1. ``style="record"`` — multi-field structured documents (richer realistic form).
   The asked field becomes one of several, so attribute-grounding is tested under
   realistic tool output, WITHOUT breaking any judge-free structural invariant.
2. ``DeliveryMode.AGENTIC_COMMIT`` — action-consequential framing where the agent
   WRITES the value (COMMIT) or hands off (ESCALATE), so we can measure *acting on*
   a mis-grounded value rather than merely reporting it.
"""

from __future__ import annotations

import re

from uptake_probe.detect import has_value_code
from uptake_probe.domains import BUSINESS_DOMAINS
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.scaffolds import run
from uptake_probe.synth import make_instance
from uptake_probe.validate import structural_issues

_CODE = re.compile(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}")
Q = ResultQuality
_CONDS = [Q.GOOD, Q.ATTR_MISMATCH, Q.IRRELEVANT, Q.IRRELEVANT_NEAR,
          Q.IRRELEVANT_FAR, Q.STALE, Q.PARTIAL]


def test_record_style_structurally_clean_all_domains():
    seed = 70000
    for dom in BUSINESS_DOMAINS:
        for q in _CONDS:
            for _ in range(10):
                inst = make_instance(seed=seed, k=5, n_necessary=1, position="center",
                                     necessary_quality=q, style="record", domain=dom)
                assert structural_issues(inst) == [], (dom, q, inst.instance_id)
                seed += 1


def test_record_good_is_multi_field():
    # a GOOD record must carry the gold PLUS at least one sibling code (so the agent
    # has to pick the right field, not copy the only code present).
    inst = make_instance(seed=71000, k=5, n_necessary=1, necessary_quality=Q.GOOD,
                         style="record", domain="payments")
    nec = next(r for r in inst.results if r.is_necessary)
    codes = set(_CODE.findall(nec.full_text))
    assert inst.gold_values[0] in codes
    assert len(codes) >= 2  # gold + at least one sibling


def test_record_attr_mismatch_omits_asked_field():
    inst = make_instance(seed=71001, k=5, n_necessary=1, necessary_quality=Q.ATTR_MISMATCH,
                         style="record", domain="logistics")
    nec = next(r for r in inst.results if r.is_necessary)
    fact = nec.fact
    assert fact.attribute not in nec.full_text       # asked field absent
    assert fact.entity in nec.full_text              # right entity (header)
    assert nec.wrong_value in nec.full_text          # decoy present as a sibling field


def test_record_partial_has_no_complete_code():
    # negative control must hold under multi-field rendering too.
    for dom in BUSINESS_DOMAINS:
        inst = make_instance(seed=71002, k=5, n_necessary=1, necessary_quality=Q.PARTIAL,
                             style="record", domain=dom)
        nec = next(r for r in inst.results if r.is_necessary)
        assert not has_value_code(nec.full_text), (dom, nec.full_text)


class _Scripted:
    """A backend that emits a fixed list of outputs, one per generate() call."""
    name = "scripted"

    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def count_tokens(self, t):
        return len(t.split())

    def generate(self, prompt):
        o = self.outs[self.i]
        self.i = min(self.i + 1, len(self.outs) - 1)
        return o


def _irrelevant_instance():
    inst = make_instance(seed=72000, k=5, n_necessary=1, necessary_quality=Q.IRRELEVANT,
                         style="record", domain="devops")
    nec = next(r for r in inst.results if r.is_necessary)
    return inst, nec.result_id, nec.wrong_value


def test_commit_writes_value():
    inst, rid, decoy = _irrelevant_instance()
    rr = run(inst, _Scripted(["Action: SEARCH[all]", f"Action: FETCH[{rid}]",
                              f"Action: COMMIT[{decoy}]"]),
             DeliveryMode.AGENTIC_COMMIT)
    assert rr.terminal == "commit"
    assert rr.answer == decoy
    assert rr.mode == DeliveryMode.AGENTIC_COMMIT.value


def test_escalate_is_safe_handoff():
    inst, rid, _ = _irrelevant_instance()
    rr = run(inst, _Scripted(["Action: SEARCH[all]", "Action: ESCALATE[different entity]"]),
             DeliveryMode.AGENTIC_COMMIT)
    assert rr.terminal == "escalate"
    assert rr.answer == ""


def test_bare_value_counts_as_commit():
    inst, rid, decoy = _irrelevant_instance()
    rr = run(inst, _Scripted(["Action: SEARCH[all]", decoy]), DeliveryMode.AGENTIC_COMMIT)
    assert rr.terminal == "commit"
    assert rr.answer == decoy


def test_report_mode_terminal_unchanged():
    # the original AGENTIC path must be untouched by the commit additions.
    inst, rid, decoy = _irrelevant_instance()
    rr = run(inst, _Scripted(["Action: SEARCH[all]", f"Action: FETCH[{rid}]",
                              f"Action: ANSWER[{decoy}]"]),
             DeliveryMode.AGENTIC)
    assert rr.terminal == "answer"
    assert rr.mode == DeliveryMode.AGENTIC.value
    assert rr.answer == decoy
