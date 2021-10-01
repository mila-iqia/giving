import pytest
from varname import ImproperUseError, VarnameRetrievingError

from giving import give, given, giver
from giving.gvr import LinePosition, register_special, resolve


def accumulate(key):
    return given().getitem(key, strict=False).values()


def bisect(arr, key):
    lo = -1
    hi = len(arr)
    while lo < hi - 1:
        mid = lo + (hi - lo) // 2
        give(mid=mid)
        if arr[mid] > key:
            hi = mid
        else:
            lo = mid
    return lo + 1


def give_arg(x):
    give(x)


def give_assign(x):
    y = give(x)
    return y


# def give_multi_assign(x):
#     z = y = give(x)
#     return y, z


def give_multi_assign2(x):
    (a, b), c = (x, x + 1), x + 2
    give()
    return a, b, c


def give_above(x):
    y = x * x
    give()
    return y


def give_above_multi_assign(x):
    z = y = x * x
    give()
    return y, z


def give_above_annassign(x):
    y: int = x * x
    give()
    return y


def give_above_augassign(x):
    x += x * x
    give()
    return x


grog = 10


def give_global():
    global grog
    grog = 15
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

    # with accumulate("y") as results:
    #     give_multi_assign(4)
    # assert results == [4]

    with given()["a", "b", "c"].values() as results:
        give_multi_assign2(4)
    assert results == [(4, 5, 6)]

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

    with accumulate("grog") as results:
        give_global()
    assert results == [grog]


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
        results = g.accum()

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
        results = g.accum()

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
        results = g.accum()

        a = 3  # noqa: F841
        giv()
        b = giv(x=5)  # noqa: F841

        assert results == [{"a": 3, "$test": 1234}, {"x": 5, "$test": 1234}]


def test_special_time():
    import time

    t = int(time.time())

    with given() as g:
        results = g["$time"].map(int).accum()

        give.time(a=4)

    assert results == [t]


def test_special_frame():
    giv = giver("$frame")

    with given() as g:
        results = g["$frame"].accum()

        giv(a=4)

    (fr,) = results
    assert fr.f_code.co_filename == __file__


def test_special_line():
    # Note: this should be equal to the line number of giv(a=4)
    lineno = test_special_line.__code__.co_firstlineno + 7

    with given() as g:
        results = g["$line"].accum()

        give.line(a=4)

    assert results == [LinePosition("test_special_line", __file__, lineno)]


def test_ksubscribe():
    with given() as g:
        results = []

        @g.ksubscribe
        def _(a=None, b=None):
            results.append((a, b))

        give(a=1, b=2)
        give(a=3)
        give(b=4)

    assert results == [(1, 2), (3, None), (None, 4)]


def test_inherit():
    with given() as g:
        results = g["a", "b"].accum()

        with give.inherit(a=1):
            give(b=2)
            give(b=3)

        assert results == [(1, 2), (1, 3)]


def test_inherit_nested():
    with given() as g:
        results = g["a", "b"].accum()

        with give.inherit(a=1):
            with give.inherit(a=5, b=6):
                give(b=2)
                give(b=3)

        assert results == [(5, 2), (5, 3)]


def test_wrap():
    with given() as g:

        resultsaw = g.where("a")["a", "w"].accum()

        resultsw = []

        @g.kwrap("w")
        def _(w):
            resultsw.append(w)
            yield
            resultsw.append(-w)

        with give.wrap_inherit("w", w=1):
            give(a=10)
            with give.wrap("w", w=2):
                give(a=20)
            give(a=30)

        assert resultsaw == [(10, 1), (20, 1), (30, 1)]
        assert resultsw == [1, 2, -2, -1]


def test_wrap2():
    from contextlib import contextmanager

    results = []

    @contextmanager
    def mng1():
        results.append("(")
        yield
        results.append(")")

    @contextmanager
    def mng2():
        results.append("<")
        yield
        results.append(">")

    with given() as g:
        g.wrap("A", mng1())
        g.wrap("B", mng2)
        g["?a"] >> results

        with give.wrap("A"):
            give(a=10)
            with give.wrap("B"):
                give(a=20)
            give(a=30)

    assert results == ["(", 10, "<", 20, ">", 30, ")"]


def test_variant():
    @give.variant
    def stuff(data):
        return {"stuff": data, "transformed": True}

    with given() as gv:
        results = gv.accum()

        x = 3
        stuff()
        y = 4
        give(y)
        stuff(x, y)

    assert results == [
        {"stuff": {"x": 3}, "transformed": True},
        {"y": 4},
        {"stuff": {"x": 3, "y": 4}, "transformed": True},
    ]


def test_variant_2():
    give2 = giver("zazz")

    @give2.variant
    def stuff(data):
        return {"stuff": data, "transformed": True}

    with given() as gv:
        results = gv.accum()

        x = stuff(3)
        y = 4
        stuff(y, z=x)

    assert results == [
        {"stuff": {"zazz": 3}, "transformed": True},
        {"stuff": {"zazz": 4, "z": 3}, "transformed": True},
    ]


def test_given_reenter():
    gv = given()
    results = gv["a"].accum()

    with gv:
        give(a=1)
        give(a=7)

    assert results == [1, 7]

    with pytest.raises(Exception):
        with gv:
            pass


def test_given_notroot():
    gv = given()

    with pytest.raises(Exception):
        with gv["?a"]:
            pass


def test_exec():
    def _main(a):
        give(x=a)
        give(x=a + 1)

    gv = given()
    results = gv["x"].accum()
    gv.exec(_main, 41)

    assert results == [41, 42]


def test_eval():
    def _main(a):
        give(x=a)
        give(x=a + 1)

    results = given()["x"].eval(_main, 41)

    assert results == [41, 42]
