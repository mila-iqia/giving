import pytest
from rx import operators as op
from varname import ImproperUseError, VarnameRetrievingError

from giving import accumulate, give, given, giver
from giving.core import register_special, resolve


def bisect(arr, key):
    lo = -1
    hi = len(arr)
    while lo < hi - 1:
        mid = lo + (hi - lo) // 2
        give(mid=mid)
        if (elem := arr[mid]) > key:
            hi = mid
        else:
            lo = mid
    return lo + 1


def give_arg(x):
    give(x)


def give_assign(x):
    y = give(x)


def give_multi_assign(x):
    z = y = give(x)


def give_above(x):
    y = x * x
    give()


def give_above_multi_assign(x):
    z = y = x * x
    give()


def give_above_annassign(x):
    y: int = x * x
    give()


def give_above_augassign(x):
    x += x * x
    give()


def _fetch(*args):
    return resolve(frame=1, func=_fetch, args=args)


def test_resolve():
    a = 3

    _ = [_fetch(a)]
    assert _[0] == {"a": a}

    z = _fetch(a)
    assert z == {"z": a}

    t = 1234
    _ = [_fetch()]
    assert _[0] == {"t": t}

    _ = _fetch(a, z, t)
    assert _ == {"a": a, "z": z, "t": t}


def test_give():
    with accumulate("x") as results:
        give_arg(3)
    assert results == [3]

    with accumulate("y") as results:
        give_assign(4)
    assert results == [4]

    with accumulate("y") as results:
        give_above(4)
    assert results == [16]

    with accumulate("y") as results:
        give_above_multi_assign(4)
    assert results == [16]

    with accumulate("y") as results:
        give_above_annassign(4)
    assert results == [16]

    with accumulate("x") as results:
        give_above_augassign(4)
    assert results == [20]


def test_give_bad():
    def bad1():
        give()

    def bad2():
        1 + 1
        give()

    def bad3():
        exec(
            """
x = 10
give()
"""
        )

    with given() as gv:
        gv.subscribe(print)

        with pytest.raises(ImproperUseError):
            bad1()

        with pytest.raises(ImproperUseError):
            bad2()

        with pytest.raises(VarnameRetrievingError):
            bad3()


def test_giver():
    giv = giver("x")

    with given() as g:
        g >> (results := [])

        giv(1)
        giv(2, y=3)

        assert results == [{"x": 1}, {"x": 2, "y": 3}]

        with pytest.raises(ImproperUseError, match="1 positional argument"):
            giv()

        with pytest.raises(ImproperUseError, match="1 positional argument"):
            giv(1, 2)


def test_giver_2():
    giv = giver("x", "y")

    with given() as g:
        g >> (results := [])

        giv(1, 2)
        giv(3, 4)

        assert results == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

        with pytest.raises(ImproperUseError, match="2 positional argument"):
            giv()

        with pytest.raises(ImproperUseError, match="2 positional argument"):
            giv(1)


@register_special("$test")
def special_test():
    return 1234


def test_giver_special():
    giv = giver("$test")

    with given() as g:
        g >> (results := [])

        a = 3
        giv()
        b = giv(x=5)

        assert results == [{"a": 3, "$test": 1234}, {"x": 5, "$test": 1234}]
