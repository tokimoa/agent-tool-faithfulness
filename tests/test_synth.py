from uptake_probe.synth import make_instance


def test_counts_and_necessary_positions():
    inst = make_instance(seed=1, k=8, n_necessary=2, position="center")
    assert inst.k == 8
    assert len(inst.results) == 8
    assert len(inst.necessary_facts) == 2
    necessary_idxs = [i for i, r in enumerate(inst.results) if r.is_necessary]
    assert necessary_idxs == inst.necessary_positions
    assert len(necessary_idxs) == 2


def test_center_position_is_central():
    inst = make_instance(seed=2, k=9, n_necessary=1, position="center")
    assert inst.necessary_positions == [4]


def test_early_and_late_positions():
    assert make_instance(seed=2, k=6, n_necessary=1, position="early").necessary_positions == [0]
    assert make_instance(seed=2, k=6, n_necessary=1, position="late").necessary_positions == [5]


def test_determinism():
    a = make_instance(seed=5, k=4, n_necessary=1)
    b = make_instance(seed=5, k=4, n_necessary=1)
    assert [r.full_text for r in a.results] == [r.full_text for r in b.results]


def test_contradictory_distractor_targets_a_necessary_fact():
    inst = make_instance(seed=7, k=4, n_necessary=1, include_contradictory=True)
    contra = inst.contradictory_results
    assert len(contra) >= 1
    c = contra[0]
    assert c.contradicts_entity == inst.necessary_facts[0].entity
    assert c.wrong_value != inst.necessary_facts[0].value


def test_authority_cue_changes_only_the_instruction():
    cued = make_instance(seed=9, k=4, n_necessary=1, include_contradictory=True,
                         authority_cue=True)
    nocue = make_instance(seed=9, k=4, n_necessary=1, include_contradictory=True,
                          authority_cue=False)
    assert "official registry" in cued.question
    assert "official registry" not in nocue.question
    # same planted facts/results regardless of the cue (only wording differs)
    assert cued.gold_values == nocue.gold_values
    assert nocue.meta["authority_cue"] is False


def test_mismatch_signal_prepends_cue_without_changing_the_decoy():
    from uptake_probe.schema import ResultQuality

    none = make_instance(seed=11, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.IRRELEVANT, mismatch_signal="none")
    hard = make_instance(seed=11, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.IRRELEVANT, mismatch_signal="hard")
    none_nec = next(r for r in none.results if r.is_necessary)
    hard_nec = next(r for r in hard.results if r.is_necessary)
    # the cue text is present only when requested...
    assert not none_nec.full_text.startswith("No exact match was found")
    assert hard_nec.full_text.startswith("No exact match was found")
    # ...and is the ONLY difference: the same seed plants the same decoy passage.
    assert none_nec.wrong_value == hard_nec.wrong_value
    assert hard_nec.full_text.endswith(none_nec.full_text)
    assert hard.meta["mismatch_signal"] == "hard"


def test_mismatch_signal_is_noop_for_good_results():
    from uptake_probe.schema import ResultQuality

    plain = make_instance(seed=12, k=4, n_necessary=1, necessary_quality=ResultQuality.GOOD)
    cued = make_instance(seed=12, k=4, n_necessary=1, necessary_quality=ResultQuality.GOOD,
                         mismatch_signal="hard")
    assert [r.full_text for r in plain.results] == [r.full_text for r in cued.results]


def test_attr_mismatch_keeps_entity_changes_attribute_with_a_decoy():
    from uptake_probe.schema import ResultQuality

    inst = make_instance(seed=21, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH)
    fact = inst.necessary_facts[0]
    nec = next(r for r in inst.results if r.is_necessary)
    # the result is about the RIGHT entity (identity check would pass)...
    assert fact.entity in nec.full_text
    # ...but about a DIFFERENT attribute, with a decoy value present.
    assert fact.attribute not in nec.full_text
    assert nec.wrong_value is not None and nec.wrong_value in nec.full_text
    assert fact.entity in inst.unobtainable_entities


def test_stale_marks_value_superseded_with_a_decoy():
    from uptake_probe.schema import ResultQuality

    inst = make_instance(seed=22, k=4, n_necessary=1, necessary_quality=ResultQuality.STALE)
    nec = next(r for r in inst.results if r.is_necessary)
    assert "superseded" in nec.full_text and "not included" in nec.full_text
    assert nec.wrong_value is not None and nec.wrong_value in nec.full_text


def test_partial_truncates_value_with_no_decoy_and_no_full_code():
    from uptake_probe.detect import has_value_code
    from uptake_probe.schema import ResultQuality

    inst = make_instance(seed=23, k=4, n_necessary=1, necessary_quality=ResultQuality.PARTIAL)
    nec = next(r for r in inst.results if r.is_necessary)
    assert "truncated" in nec.full_text
    assert nec.wrong_value is None              # nothing complete to copy
    assert not has_value_code(nec.full_text)    # no full value-code leaked into the result


def test_realistic_style_preserves_nonce_semantics_but_changes_surface():
    from uptake_probe.schema import ResultQuality

    tmpl = make_instance(seed=31, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH, style="template")
    real = make_instance(seed=31, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH, style="realistic")
    t_nec = next(r for r in tmpl.results if r.is_necessary)
    r_nec = next(r for r in real.results if r.is_necessary)
    # same planted facts + same decoy (judge-free detection unchanged)...
    assert tmpl.gold_values == real.gold_values
    assert t_nec.wrong_value == r_nec.wrong_value
    assert r_nec.wrong_value in r_nec.full_text          # nonce decoy still verbatim
    # ...but a different (non-template) surface form
    assert r_nec.full_text != t_nec.full_text
    assert not r_nec.full_text.startswith("The ")        # not the rigid template opener
    assert real.meta["style"] == "realistic"
    assert "realistic" in real.instance_id


def test_realistic_stale_and_partial_keep_recognition_markers():
    from uptake_probe.detect import detect_abstention, has_value_code
    from uptake_probe.schema import ResultQuality

    stale = make_instance(seed=32, k=4, n_necessary=1,
                          necessary_quality=ResultQuality.STALE, style="realistic")
    s_nec = next(r for r in stale.results if r.is_necessary)
    assert detect_abstention(s_nec.full_text)            # superseded/rotated/not included present
    assert s_nec.wrong_value in s_nec.full_text

    part = make_instance(seed=33, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.PARTIAL, style="realistic")
    p_nec = next(r for r in part.results if r.is_necessary)
    assert p_nec.wrong_value is None
    assert not has_value_code(p_nec.full_text)           # no complete value leaked


def test_rejects_n_greater_than_k():
    import pytest

    with pytest.raises(ValueError):
        make_instance(seed=1, k=2, n_necessary=3)
