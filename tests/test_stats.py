from uptake_probe.stats import bootstrap_ci, rate_ci, wilson_ci


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_degenerate_cases_stay_in_unit_interval():
    import pytest

    assert wilson_ci(0, 20)[0] == 0.0                    # all-zero: lower bound pinned at 0
    assert wilson_ci(20, 20)[1] == pytest.approx(1.0)    # all-one: upper bound ~1 (FP)
    assert wilson_ci(0, 0) == (0.0, 1.0)                 # no data: maximally uncertain


def test_wilson_narrows_as_n_grows():
    lo10, hi10 = wilson_ci(5, 10)
    lo100, hi100 = wilson_ci(50, 100)
    assert (hi100 - lo100) < (hi10 - lo10)


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    bits = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    a = bootstrap_ci(bits, seed=0)
    b = bootstrap_ci(bits, seed=0)
    assert a == b                               # reproducible
    assert a[0] <= 0.6 <= a[1]


def test_rate_ci_shape():
    d = rate_ci([True, False, True, True])
    assert d["n"] == 4 and d["k"] == 3 and d["rate"] == 0.75
    assert 0.0 <= d["ci_lo"] <= 0.75 <= d["ci_hi"] <= 1.0
