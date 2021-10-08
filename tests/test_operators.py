import pytest

from giving import give, given, giver, operators as op

TOLERANCE = 1e-6


def fib(n):
    a = 0
    b = 1
    give(a, b)
    for _ in range(n - 1):
        a, b = b, a + b
        give(a, b)
    return b


def things(*values):
    for a in values:
        give(a)


def test_getitem():
    with given() as gv:
        results = []
        gv.pipe(op.getitem("b")).subscribe(results.append)
        fib(5)
        assert results == [1, 1, 2, 3, 5]


def test_getitem2():
    with given() as gv:
        results = []
        gv.pipe(op.getitem("a", "b")).subscribe(results.append)
        fib(5)
        assert results == [(0, 1), (1, 1), (1, 2), (2, 3), (3, 5)]


def test_getitem_tuple():
    with given() as gv:
        results = gv.kmap(lambda a: (a, a * a))[1].accum()

        things(1, 2, 3)
        assert results == [1, 4, 9]


def test_getitem_strict():
    with given() as gv:
        results = []
        gv.pipe(op.getitem("a", "b", strict=True)).subscribe(results.append)
        fib(5)
        with pytest.raises(KeyError):
            give(a=123)


def test_format():
    with given() as gv:
        results = []
        gv.pipe(op.format("b={b}")).subscribe(results.append)
        fib(5)
        assert results == ["b=1", "b=1", "b=2", "b=3", "b=5"]


def test_format2():
    with given() as gv:
        results = []
        gv.pipe(op.getitem("b"), op.format("b={}")).subscribe(results.append)
        fib(5)
        assert results == ["b=1", "b=1", "b=2", "b=3", "b=5"]


def test_format3():
    with given() as gv:
        results = []
        gv.pipe(op.getitem("a", "b"), op.format("a={},b={}")).subscribe(results.append)
        fib(5)
        assert results == [
            "a=0,b=1",
            "a=1,b=1",
            "a=1,b=2",
            "a=2,b=3",
            "a=3,b=5",
        ]


def test_format4():
    with given() as gv:
        results = gv.format("{a} {b}", skip_missing=True).accum()

        give(a=1)
        give(b=2)
        give(a=3, b=4)

        assert results == ["3 4"]


def test_format_raw():
    with given() as gv:
        results = gv["a", "b"].format("{}", raw=True).accum()

        give(a=1, b=2)
        give(a=3, b=4)

        assert results == ["(1, 2)", "(3, 4)"]


def test_kmap():
    with given() as gv:
        results = []
        gv.pipe(op.kmap(lambda b: -b)).subscribe(results.append)
        fib(5)
        assert results == [-1, -1, -2, -3, -5]


def test_kmap2():
    with given() as gv:
        results = gv.kmap(x=lambda **kw: -kw["b"], y=lambda a: a * a).accum()
        fib(5)
        assert results == [
            {"x": -1, "y": 0},
            {"x": -1, "y": 1},
            {"x": -2, "y": 1},
            {"x": -3, "y": 4},
            {"x": -5, "y": 9},
        ]


def test_kmap_err():
    with given() as gv:
        with pytest.raises(TypeError):
            gv.kmap(lambda a: -a, b=lambda b: -b)


def test_augment():
    with given() as gv:
        results = gv.augment(x=lambda **kw: -kw["b"], y=lambda a: a * a).accum()
        fib(5)
        assert results == [
            {"a": 0, "b": 1, "x": -1, "y": 0},
            {"a": 1, "b": 1, "x": -1, "y": 1},
            {"a": 1, "b": 2, "x": -2, "y": 1},
            {"a": 2, "b": 3, "x": -3, "y": 4},
            {"a": 3, "b": 5, "x": -5, "y": 9},
        ]


def test_augment_overwrite():
    with given() as gv:
        results = gv.kmap(a=lambda **kw: -kw["b"], b=lambda a: a * a).accum()
        fib(5)
        assert results == [
            {"a": -1, "b": 0},
            {"a": -1, "b": 1},
            {"a": -2, "b": 1},
            {"a": -3, "b": 4},
            {"a": -5, "b": 9},
        ]


def test_kfilter():
    with given() as gv:
        results = gv.kfilter(lambda a: a > 0)["a"].accum()

        things(0, 1, -2, 3, -4, 5)

        assert results == [1, 3, 5]


def test_roll():
    with given() as gv:
        results = gv["b"].roll(3).map(list).accum()
        fib(5)
        assert results == [[1], [1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 5]]


def test_rolling_average():
    with given() as gv:
        results1 = []
        results2 = []
        bs = gv.pipe(op.getitem("b"))

        bs.pipe(
            op.average(scan=7),
        ).subscribe(results1.append)

        bs.pipe(
            op.roll(7),
            op.map(lambda xs: sum(xs) / len(xs)),
        ).subscribe(results2.append)

        fib(25)
        assert all(abs(m1 - m2) < TOLERANCE for m1, m2 in zip(results1, results2))


def test_rolling_average_and_variance():
    with given() as gv:
        bs = gv.pipe(op.getitem("b"))

        results1 = []
        bs.pipe(
            op.average_and_variance(scan=7),
            op.skip(1),
        ).subscribe(results1.append)

        def meanvar(xs):
            n = len(xs)
            if len(xs) >= 2:
                mean = sum(xs) / n
                var = sum((x - mean) ** 2 for x in xs) / (n - 1)
                return (mean, var)
            else:
                return (None, None)

        results2 = []
        bs.pipe(
            op.roll(7),
            op.map(meanvar),
            op.skip(1),
        ).subscribe(results2.append)

        fib(25)
        assert all(
            abs(m1 - m2) < TOLERANCE and abs(v1 - v2) < TOLERANCE
            for (m1, v1), (m2, v2) in zip(results1, results2)
        )


def test_variance():
    with given() as gv:
        bs = gv.pipe(op.getitem("b"))

        results1 = []
        bs.pipe(
            op.variance(scan=7),
            op.skip(1),
        ).subscribe(results1.append)

        def varcalc(xs):
            n = len(xs)
            if len(xs) >= 2:
                mean = sum(xs) / n
                var = sum((x - mean) ** 2 for x in xs) / (n - 1)
                return var
            else:
                return (None, None)

        results2 = []
        bs.pipe(
            op.roll(7),
            op.map(varcalc),
            op.skip(1),
        ).subscribe(results2.append)

        fib(25)
        assert all(abs(v1 - v2) < TOLERANCE for v1, v2 in zip(results1, results2))


def accum(obs):
    results = []
    obs.subscribe(results.append)
    return results


def test_average():
    values = [1, 2, 10, 20]

    with given() as gv:
        gv = gv.pipe(op.getitem("a"))

        results1 = []
        gv.pipe(op.average()).subscribe(results1.append)

        results2 = []
        gv.pipe(op.average(scan=True)).subscribe(results2.append)

        results3 = []
        gv.pipe(op.average(scan=2)).subscribe(results3.append)

        things(*values)

    assert results1 == [sum(values) / len(values)]
    assert results2 == [
        sum(values[:i]) / len(values[:i]) for i in range(1, len(values) + 1)
    ]
    assert results3 == [values[0]] + [
        (a + b) / 2 for a, b in zip(values[:-1], values[1:])
    ]


def test_count():
    values = [1, 3, -4, 21, -8, -13]

    with given() as gv:
        gv = gv.pipe(op.getitem("a"))

        results1 = []
        gv.pipe(op.count()).subscribe(results1.append)

        results2 = []
        gv.pipe(op.count(lambda x: x > 0)).subscribe(results2.append)

        results3 = []
        gv.pipe(op.count(lambda x: x > 0, scan=True)).subscribe(results3.append)

        results4 = []
        gv.pipe(op.count(lambda x: x > 0, scan=3)).subscribe(results4.append)

        results5 = []
        gv.pipe(op.count(scan=True)).subscribe(results5.append)

        results6 = []
        gv.pipe(op.count(scan=3)).subscribe(results6.append)

        things(*values)

    assert results1 == [len(values)]
    assert results2 == [len([v for v in values if v > 0])]
    assert results3 == [1, 2, 2, 3, 3, 3]
    assert results4 == [1, 2, 2, 2, 1, 1]
    assert results5 == list(range(1, len(values) + 1))
    assert results6 == [1, 2, 3, 3, 3, 3]


def test_min():
    values = [1, 3, -4, 21, -8, -13]

    with given() as gv:
        results = gv["a"].min().accum()

        things(*values)

    assert results == [-13]


def test_max():
    values = [1, 3, -4, 21, -8, -13]

    with given() as gv:
        results = gv["a"].max().accum()

        things(*values)

    assert results == [21]


def test_min_cmp():
    values = [1, 3, -4, 21, -8, -30]

    with given() as gv:
        results = gv["a"].min(comparer=lambda x, y: abs(x) - abs(y)).accum()

        things(*values)

    assert results == [1]


def test_max_cmp():
    values = [1, 3, -4, 21, -8, -30]

    with given() as gv:
        results = gv["a"].max(comparer=lambda x, y: abs(x) - abs(y)).accum()

        things(*values)

    assert results == [-30]


def test_min_key():
    values = [1, 3, -4, 21, -8, -30]

    with given() as gv:
        results = gv["a"].min(key=abs).accum()

        things(*values)

    assert results == [1]


def test_max_key():
    values = [1, 3, -4, 21, -8, -30]

    with given() as gv:
        results = gv["a"].max(key=abs).accum()

        things(*values)

    assert results == [-30]


def test_min_key2():
    values = [1, 3, -4, 21, -8, -30]

    with given() as gv:
        results = gv.min(key="a").accum()

        things(*values)

    assert results == [{"a": -30}]


def test_sum():
    values = [1, 3, -4, 21, -8, -17]

    with given() as gv:
        results = gv["a"].sum().accum()

        things(*values)

    assert results == [-4]


def test_affix():
    values = [1, 2, 3, 4]

    with given() as gv:
        results = []
        gv.pipe(
            op.affix(b=gv.pipe(op.getitem("a"), op.map(lambda x: x * x)))
        ).subscribe(results.append)

        things(*values)

    assert results == [{"a": x, "b": x * x} for x in values]


def test_affix2():
    values = [1, 2, 3, 4]

    with given() as gv:
        results = gv.where("a").affix(asquare=lambda o: o.kmap(lambda a: a * a)).accum()

        give(b=7)
        things(*values)

    assert results == [{"a": x, "asquare": x * x} for x in values]


def varia():
    give(x=1)
    give(x=2, y=True)
    give(z=100)
    give(x=3, y=False)
    give(y=True)


def test_where():
    with given() as gv:
        everything = accum(gv)

        results1 = accum(gv.pipe(op.where("x")))
        results2 = accum(gv.pipe(op.where(y=True)))
        results3 = accum(gv.pipe(op.where("x", y=True)))
        results4 = accum(gv.pipe(op.where(x=lambda x: x > 10)))
        results5 = accum(gv.pipe(op.where("!x")))
        results6 = accum(gv.pipe(op.where(x=2, y=True)))

        varia()

    assert results1 == [d for d in everything if "x" in d]
    assert results2 == [d for d in everything if "y" in d and d["y"]]
    assert results3 == [d for d in everything if "x" in d and "y" in d and d["y"]]
    assert results4 == [d for d in everything if "x" in d and d["x"] > 10]
    assert results5 == [d for d in everything if "x" not in d]
    assert results6 == [{"x": 2, "y": True}]


def test_where_any():
    with given() as gv:
        everything = gv.accum()
        results = gv.where_any("x", "z").accum()

        varia()

    assert results == [d for d in everything if "x" in d or "z" in d]


def aggron(n):
    for i in range(n):
        give(start=True)
        give(a=i)
        give(b=i * i)
        give(end=True)


def test_collect_between():
    with given() as gv:
        results = gv.pipe(op.collect_between("start", "end")).accum()

        aggron(3)

    assert results == [
        {"start": True, "end": True, "a": 0, "b": 0},
        {"start": True, "end": True, "a": 1, "b": 1},
        {"start": True, "end": True, "a": 2, "b": 4},
    ]


def fact(n):
    giv = giver(n=n)
    giv(start=True)
    give(dummy=1234)
    if n <= 1:
        value = n
    else:
        f1 = giv(fact(n - 1))
        value = n * f1
    return giv(value)


def test_collect_between2():
    with given() as gv:
        results = gv.pipe(op.collect_between("begin", "value", common="n")).accum()

        fact(6)

    print(results)

    assert results == [
        {"start": True, "n": 1, "value": 1},
        {"start": True, "n": 2, "value": 2, "f1": 1},
        {"start": True, "n": 3, "value": 6, "f1": 2},
        {"start": True, "n": 4, "value": 24, "f1": 6},
        {"start": True, "n": 5, "value": 120, "f1": 24},
        {"start": True, "n": 6, "value": 720, "f1": 120},
    ]


def test_as():
    with given() as gv:
        results = gv["a"].as_("z").accum()

        things(1, 2)

        assert results == [{"z": 1}, {"z": 2}]


def test_kmerge():
    with given() as gv:
        results1 = gv.kmerge().accum()
        results2 = gv.kscan().accum()

        give(a=1)
        give(b=2)
        give(b=3, c=4)

    assert results1 == [{"a": 1, "b": 3, "c": 4}]

    assert results2 == [
        {"a": 1},
        {"a": 1, "b": 2},
        {"a": 1, "b": 3, "c": 4},
    ]


def test_keep():
    with given() as gv:
        results = gv.keep(a="z").accum()

        things(1, 2)

        assert results == [{"z": 1}, {"z": 2}]


def test_keep_2():
    with given() as gv:
        results = gv.keep("b", c="d").accum()

        give(a=1, b=2, c=3)
        give(a=4, b=5, c=6)

        assert results == [{"b": 2, "d": 3}, {"b": 5, "d": 6}]


def test_tag():
    with given() as gv:
        results = gv.tag(group="anemone").accum()

        things(1, 2, 3)

    assert results == [
        {"a": 1, "$group": "anemone", "$word": "share"},
        {"a": 2, "$group": "anemone", "$word": "hope"},
        {"a": 3, "$group": "anemone", "$word": "push"},
    ]


def test_tag2():
    class X:
        pass

    with given() as gv:
        results = gv["a"].tag(group="gargamel", field="wrrd").accum()

        things(X(), X())

    x0, x1 = results

    assert x0.wrrd == "fill"
    assert x1.wrrd == "stay"

    assert getattr(x0, "$word") == "fill"
    assert getattr(x1, "$word") == "stay"

    assert getattr(x0, "$group") == "gargamel"
    assert getattr(x1, "$group") == "gargamel"


def test_group_wrap():
    with given() as g:
        results = []
        results2 = []

        gw = g.group_wrap("A")
        gw.subscribe(lambda grp: grp["a"].sum() >> results)

        gw = g.group_wrap("B")
        gw.subscribe(lambda grp: grp["a"].sum() >> results2)

        with give.wrap("A"):
            give(a=1)
            give(a=2)

        give(a=3)

        with give.wrap("B"):
            give(a=4)
            give(a=5)

        assert results == [3]
        assert results2 == [9]


def test_sole():
    with given() as gv:
        results = gv.sole().accum()
        results2 = gv.sole(keep_key=True).accum()

        give(x=1)
        give(y=2)
        give(z=3)

    assert results == [1, 2, 3]
    assert results2 == [("x", 1), ("y", 2), ("z", 3)]


def test_sole_exclude():
    with given() as gv:
        results = gv.sole(exclude="woo").accum()
        results2 = gv.sole(exclude=["y"]).accum()

        give(x=1)
        give(y=2, woo=77)
        give(z=3)

    assert results == [1, 2, 3]
    assert results2 == [1, 77, 3]


def test_sole_errors():
    with given() as gv:
        gv.sole().display()

        with pytest.raises(Exception):
            give(x=1, y=2)

    with given() as gv:
        gv.sole(exclude="z").display()

        with pytest.raises(Exception):
            give(z=3)


def test_bottom():
    with given() as gv:
        results1 = gv["a"].bottom(n=3).accum()
        results2 = gv["a"].bottom(n=3, key=abs).accum()
        results3 = gv.bottom(n=2, key="a").accum()

        things(1, 5, 2, 99, 3, -7, 21)

    assert results1 == [-7, 1, 2]
    assert results2 == [1, 2, 3]
    assert results3 == [{"a": -7}, {"a": 1}]


def test_top():
    with given() as gv:
        results1 = gv["a"].top(n=3).accum()
        results2 = gv["a"].top(n=3, key=abs).accum()

        things(1, 5, 2, 99, 3, -7, 21)

    assert results1 == [99, 21, 5]
    assert results2 == [99, 21, -7]


def test_bottom_empty():
    with given() as gv:
        results1 = gv["a"].top(n=3).accum()

    assert results1 == []
