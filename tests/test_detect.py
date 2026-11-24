from uptake_probe.detect import (
    detect_abstention,
    detect_uptake,
    detect_value,
    fact_in_text,
    followed_contradiction,
    ngram_overlap,
)
from uptake_probe.schema import NonceFact
from uptake_probe.synth import make_instance


def test_detect_value_surface_forms():
    assert detect_value("AB23-CD45", "the code is ab23-cd45 ok")
    assert detect_value("AB23-CD45", "code AB23CD45 here")   # hyphen dropped
    assert not detect_value("AB23-CD45", "no code present")


def test_ngram_overlap():
    s = "the activation code for the foo entity is here today"
    assert ngram_overlap(s, "junk " + s + " more", n=5) == 1.0
    assert ngram_overlap(s, "totally unrelated string", n=5) == 0.0


def test_fact_in_text_by_value():
    f = NonceFact("Foo-1 protocol", "access key", "QQ22-WW33")
    assert fact_in_text(f, "the answer is QQ22-WW33")
    assert not fact_in_text(f, "nothing relevant here")


def test_detect_abstention_recognises_staleness_and_unavailability():
    # genuine mis-grounding: presents a value as the answer with no flag -> NOT abstention
    assert not detect_abstention("The access key for the Marneth-5641 module is KM54-PQ25.")
    # transparent staleness flagging -> abstention (recognised the current value is absent)
    assert detect_abstention("The current access key is not included; the previous key was NF94-MY32.")
    assert detect_abstention("The access key is KM54-PQ25 (superseded). The current key is not provided.")
    assert detect_abstention("This value is no longer current and has been rotated out.")
    # recognition phrasings found by the human-validation pass (96.8% agreement)
    assert detect_abstention("the calibration code is not currently available in the registry")
    assert detect_abstention("the access key is not explicitly stated in the provided results")
    assert detect_abstention("it has since been rotated and is no longer included in the record")
    # existing phrasings still work
    assert detect_abstention("the value is not available in the results")
    assert detect_abstention("the result only mentions a different entity")
    # a bare assertion of a value as the answer is STILL a confabulation (not abstention)
    assert not detect_abstention("The activation code for the Vexmar-2367 consortium is TJ48-MD57.")
    assert not detect_abstention("QN73-DS28")


def test_detect_uptake_answer_and_followed_contradiction():
    inst = make_instance(seed=11, k=3, n_necessary=1, include_contradictory=True)
    fact = inst.necessary_facts[0]
    wrong = inst.contradictory_results[0].wrong_value

    up = detect_uptake(inst, answer=f"{fact.entity} = {fact.value}", trace="")
    assert up[fact.entity]["answer"] is True

    foll = followed_contradiction(inst, answer=f"the value is {wrong}")
    assert foll[fact.entity] is True
    # answering correctly should NOT count as following the contradiction
    assert followed_contradiction(inst, answer=fact.value)[fact.entity] is False
