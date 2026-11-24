"""Plumbing tests for the multi-agent harness (MockModel; structure only —
meaningful fidelity numbers require a real model)."""

from uptake_probe.models import MockModel
from uptake_probe.multiagent import (
    pipeline_fidelity,
    run_pipeline,
    run_supervised,
)
from uptake_probe.schema import ResultQuality
from uptake_probe.synth import make_instance


def test_pipeline_structure():
    inst = make_instance(seed=1, k=6, n_necessary=2, position="center")
    roles = ["retriever", "relay", "answerer"]
    backends = [MockModel(), MockModel(), MockModel()]
    hops = run_pipeline(inst, roles, backends)
    assert [h.role for h in hops] == roles
    for h in hops:
        assert set(h.facts_present) == {f.entity for f in inst.necessary_facts}
    fid = pipeline_fidelity([hops])
    assert len(fid) == len(roles)
    assert all(0.0 <= c["fact_survival"] <= 1.0 for c in fid)


def test_supervision_result_fields():
    inst = make_instance(seed=2, k=5, n_necessary=1, position="center",
                         necessary_quality=ResultQuality.IRRELEVANT)
    res = run_supervised(inst, MockModel(), MockModel())
    assert isinstance(res.worker_confabulated, bool)
    assert isinstance(res.flagged, bool)
    assert isinstance(res.caught, bool)
    assert isinstance(res.false_alarm, bool)
