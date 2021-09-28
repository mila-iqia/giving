import pytest
from rx import of

from giving import give, given
from giving.gvn import Failure, ObservableProxy

from .test_operators import things


def test_proxy():
    results = []
    obs = ObservableProxy(of(1, 2, 3))
    obs.map(lambda x: x * x).map(lambda x: -x).subscribe(results.append)
    assert results == [-1, -4, -9]


def liesse():
    give(a=1)
    give(b=2)
    give(a=3, b=4)


def test_proxy_dicts():
    with given() as obs:
        results_a = []
        obs["?a"].subscribe(results_a.append)

        results_b = []
        obs["?b"].map(lambda x: x * x).subscribe(results_b.append)

        liesse()

        assert results_a == [1, 3]
        assert results_b == [4, 16]


def test_proxy_dicts_strict():
    with given() as obs:
        results_a = []
        obs["a"].subscribe(results_a.append)

        with pytest.raises(KeyError):
            liesse()


def test_give_method():
    with given() as gv:
        results1 = gv["?z1"].accum()
        results2 = gv["?z2"].accum()
        results3 = gv.where(q=True)["z3"].accum()
        results4 = gv["?z4"].accum()
        results5 = gv["?z5"].accum()

        gv["?a"].map(lambda x: x + 1).as_("z1").give()
        gv["?a"].map(lambda x: x + 1).give("z2")
        gv["?a"].map(lambda x: x + 1).as_("z3").give(q=True)
        gv["?a"].map(lambda x: (x + 2, x + 3)).give("z4", "z5")

        things(1, 2, 3)

        assert results1 == results2 == results3 == [2, 3, 4]
        assert results4 == [3, 4, 5]
        assert results5 == [4, 5, 6]


def test_rshift():
    results1 = []
    results2 = []
    results3 = set()

    with given() as gv:
        gv["a"] >> results1
        gv["a"] >> (lambda x: results2.insert(0, x))
        gv["a"] >> results3

        things(11, 22, 11, 33)

    assert results1 == [11, 22, 11, 33]
    assert results2 == [33, 11, 22, 11]
    assert results3 == {11, 22, 33}


def test_accum():
    with given() as gv:
        results1 = gv["a"].accum()
        results2 = gv["a"].accum([55])
        results3 = gv["a"].accum(set())
        with pytest.raises(TypeError):
            gv["a"].accum(())

        things(11, 22, 11, 33)

    assert results1 == [11, 22, 11, 33]
    assert results2 == [55, 11, 22, 11, 33]
    assert results3 == {11, 22, 33}


def test_fail():
    with given() as gv:
        gv["a"].filter(lambda a: a == 2).fail()

        with pytest.raises(Failure):
            things(1, 2, 3)

    with given() as gv:
        gv["a"].filter(lambda a: a == 2).fail(ValueError)

        with pytest.raises(ValueError):
            things(1, 2, 3)

    with given() as gv:
        gv["a"].filter(lambda a: a == 1234).fail()

        things(1, 2, 3)
        things(4, 5, 6)


def test_fail_if_empty():
    with pytest.raises(Failure):
        with given() as gv:
            gv["?b"].fail_if_empty()
            things(1, 2, 3)

    with given() as gv:
        gv["?a"].fail_if_empty()
        things(1, 2, 3)


def test_merge():
    with given() as gv:
        results = (gv["?a"] | gv["?b"] | gv["?c"]).accum()

        give(a=1)
        give(a=2, b=3)
        give(b=4)
        give(a=5)
        give(c=6)
        give(d=7)

    assert results == [1, 2, 3, 4, 5, 6]
