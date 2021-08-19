import traceback

import pytest
from rx import of

from giving import give, given, operators as op
from giving.obs import ObservableProxy


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
