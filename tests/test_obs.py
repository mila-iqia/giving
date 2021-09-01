import pytest
from rx import of

from giving import give, given
from giving.obs import ObservableProxy

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
        gv["?z1"] >> (results1 := [])
        gv["?z2"] >> (results2 := [])
        gv.where(q=True)["z3"] >> (results3 := [])
        gv["?z4"] >> (results4 := [])
        gv["?z5"] >> (results5 := [])

        gv["?a"].map(lambda x: x + 1).as_("z1").give()
        gv["?a"].map(lambda x: x + 1).give("z2")
        gv["?a"].map(lambda x: x + 1).as_("z3").give(q=True)
        gv["?a"].map(lambda x: (x + 2, x + 3)).give("z4", "z5")

        things(1, 2, 3)

        assert results1 == results2 == results3 == [2, 3, 4]
        assert results4 == [3, 4, 5]
        assert results5 == [4, 5, 6]
