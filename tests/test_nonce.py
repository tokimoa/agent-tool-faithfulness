import random
import re

from uptake_probe.nonce import different_value, make_nonce_fact, nonce_value


def test_value_format():
    v = nonce_value(random.Random(0))
    assert re.fullmatch(r"[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}", v)


def test_determinism():
    assert nonce_value(random.Random(42)) == nonce_value(random.Random(42))


def test_fact_fields_self_consistent():
    f = make_nonce_fact(random.Random(1))
    assert f.value in f.carrier_sentence
    assert f.entity in f.question
    assert f.attribute in f.question


def test_different_value_is_distinct():
    assert different_value(random.Random(3), "AB23-CD45") != "AB23-CD45"
