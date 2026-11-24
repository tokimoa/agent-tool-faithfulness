from uptake_probe.schema import ResultQuality
from uptake_probe.synth import make_instance
from uptake_probe.validate import is_clean, structural_issues

ALL_Q = [
    ResultQuality.GOOD, ResultQuality.IRRELEVANT, ResultQuality.IRRELEVANT_NEAR,
    ResultQuality.IRRELEVANT_FAR, ResultQuality.ATTR_MISMATCH, ResultQuality.STALE,
    ResultQuality.PARTIAL, ResultQuality.ERROR, ResultQuality.EMPTY,
    ResultQuality.REDACTED,
]


def test_generated_instances_are_structurally_clean_both_styles():
    for style in ("template", "realistic"):
        for q in ALL_Q:
            for seed in range(20):
                inst = make_instance(seed=1000 + seed, k=5, n_necessary=1,
                                     necessary_quality=q, style=style)
                issues = structural_issues(inst)
                assert not issues, f"{style}/{q.value}/seed{seed}: {issues}"


def test_validator_catches_a_leaked_true_value():
    inst = make_instance(seed=7, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH)
    nec = next(r for r in inst.results if r.is_necessary)
    # inject the true value into the mismatch result -> should be flagged
    nec.full_text += f" (leak {inst.necessary_facts[0].value})"
    assert not is_clean(inst)
    assert any("leaks the TRUE value" in s for s in structural_issues(inst))


def test_validator_catches_partial_leaking_a_full_code():
    inst = make_instance(seed=8, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.PARTIAL)
    nec = next(r for r in inst.results if r.is_necessary)
    nec.full_text += " full code AB12-CD34"
    assert any("PARTIAL leaks a complete value-code" in s for s in structural_issues(inst))
