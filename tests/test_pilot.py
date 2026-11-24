"""Integration tests: scaffolds + metrics + pilot, driven by the MockModel
simulator. These assert the *instrument discriminates* (large, deterministic,
seeded effects) — not any scientific claim, which requires a real model.
"""

from uptake_probe.models import MockModel
from uptake_probe.pilot import run_pilot


def _cell(report, mode, k):
    return next(c for c in report["cells"] if c["mode"] == mode and c["k"] == k)


def test_k_effect_is_monotone_and_large():
    rep = run_pilot(MockModel(), ks=(1, 4, 8), modes=("passive", "agentic"), n_instances=80)
    for mode in ("passive", "agentic"):
        u1 = _cell(rep, mode, 1)["uptake_rate"]
        u4 = _cell(rep, mode, 4)["uptake_rate"]
        u8 = _cell(rep, mode, 8)["uptake_rate"]
        assert u1 > u4 > u8, (mode, u1, u4, u8)
        assert u1 - u8 > 0.2, (mode, u1, u8)


def test_omission_is_agentic_only():
    rep = run_pilot(MockModel(), ks=(8,), modes=("passive", "agentic"), n_instances=80)
    # passive delivers everything in-context -> structurally no omission
    assert _cell(rep, "passive", 8)["ond_share"]["omission"] == 0.0
    # agentic must fetch -> omission is a real, large failure mode
    assert _cell(rep, "agentic", 8)["ond_share"]["omission"] > 0.2


def test_contradiction_following_fires():
    rep = run_pilot(MockModel(), ks=(4,), modes=("passive", "agentic"),
                    n_instances=80, include_contradictory=True)
    for mode in ("passive", "agentic"):
        c = _cell(rep, mode, 4)
        assert c["contradiction_following_rate"] is not None
        assert c["contradiction_following_rate"] > 0.0
        assert c["ond_share"]["distraction"] > 0.0


def test_runs_terminate_and_record_cost():
    rep = run_pilot(MockModel(), ks=(4,), modes=("agentic",), n_instances=10)
    c = _cell(rep, "agentic", 4)
    assert c["avg_tokens"] > 0
    assert c["avg_steps"] >= 2.0  # at least SEARCH + ANSWER


def test_coverage_drops_with_multiple_necessary_facts():
    # all-of-N coverage is conjunctive -> must be <= single-fact uptake
    single = run_pilot(MockModel(), ks=(6,), modes=("passive",), n_instances=80,
                       n_necessary=1)
    multi = run_pilot(MockModel(), ks=(6,), modes=("passive",), n_instances=80,
                      n_necessary=3)
    assert _cell(multi, "passive", 6)["coverage"] < _cell(single, "passive", 6)["coverage"]
