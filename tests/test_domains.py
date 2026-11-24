import re

from uptake_probe.domains import BUSINESS_DOMAINS, DOMAINS, get_domain
from uptake_probe.schema import ResultQuality
from uptake_probe.synth import make_instance
from uptake_probe.validate import structural_issues

_NONCE = re.compile(r"^[A-Z]{2}[0-9]{2}-[A-Z]{2}[0-9]{2}$")
_ALL_Q = [ResultQuality.GOOD, ResultQuality.IRRELEVANT, ResultQuality.ATTR_MISMATCH,
          ResultQuality.STALE, ResultQuality.PARTIAL]


def test_access_is_default_and_present():
    assert DOMAINS[0] == "access"
    assert set(BUSINESS_DOMAINS) == {"logistics", "payments", "devops", "support"}


def test_business_domains_generate_clean_instances_both_styles():
    for dom in BUSINESS_DOMAINS:
        for style in ("template", "realistic"):
            for q in _ALL_Q:
                for seed in range(10):
                    inst = make_instance(seed=40000 + seed, k=5, n_necessary=1,
                                         necessary_quality=q, style=style, domain=dom)
                    assert not structural_issues(inst), (dom, style, q.value, seed)


def test_values_stay_nonce_codes_across_domains():
    """The judge-free guarantee: the planted value is always an un-guessable code,
    never a domain word that a model could produce from priors."""
    for dom in BUSINESS_DOMAINS:
        for seed in range(10):
            inst = make_instance(seed=50000 + seed, k=5, n_necessary=1, domain=dom)
            for v in inst.gold_values:
                assert _NONCE.match(v), f"{dom}: non-nonce value {v!r}"


def test_attr_mismatch_keeps_entity_uses_domain_attribute():
    dom = get_domain("logistics")
    inst = make_instance(seed=3, k=4, n_necessary=1,
                         necessary_quality=ResultQuality.ATTR_MISMATCH,
                         style="realistic", domain="logistics")
    fact = inst.necessary_facts[0]
    nec = next(r for r in inst.results if r.is_necessary)
    assert fact.entity in nec.full_text          # right entity...
    assert fact.attribute not in nec.full_text   # ...wrong attribute
    assert fact.attribute in dom.attributes
    assert inst.meta["domain"] == "logistics"


def test_unknown_domain_raises():
    import pytest
    with pytest.raises(ValueError):
        make_instance(seed=1, k=4, n_necessary=1, domain="nope")
