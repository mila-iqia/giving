from give import give, given, operators as op

TOLERANCE = 1e-6


def fib(n):
    a = 0
    b = 1
    give(a, b)
    for _ in range(n - 1):
        a, b = b, a + b
        give(a, b)
    return b


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
        gv.pipe(op.getitem("a", "b"), op.format("a={},b={}")).subscribe(
            results.append
        )
        fib(5)
        assert results == [
            "a=0,b=1",
            "a=1,b=1",
            "a=1,b=2",
            "a=2,b=3",
            "a=3,b=5",
        ]


def test_keymap():
    with given() as gv:
        results = []
        gv.pipe(op.keymap(lambda b: -b)).subscribe(results.append)
        fib(5)
        assert results == [-1, -1, -2, -3, -5]


def test_roll():
    with given() as gv:
        results = []
        gv.pipe(
            op.getitem("b"),
            op.roll(3),
            op.map(list),
        ).subscribe(results.append)
        fib(5)
        assert results == [[1], [1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 5]]


def test_rolling_average():
    with given() as gv:
        results1 = []
        results2 = []
        bs = gv.pipe(op.getitem("b"))

        bs.pipe(
            op.average(roll=7),
        ).subscribe(results1.append)

        bs.pipe(
            op.roll(7),
            op.map(lambda xs: sum(xs) / len(xs)),
        ).subscribe(results2.append)

        fib(25)
        assert all(
            abs(m1 - m2) < TOLERANCE for m1, m2 in zip(results1, results2)
        )


def test_rolling_average_and_variance():
    with given() as gv:
        results1 = []
        results2 = []
        bs = gv.pipe(op.getitem("b"))

        bs.pipe(
            op.average_and_variance(roll=7),
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


def accum(obs):
    results = []
    obs.subscribe(results.append)
    return results


def things(*values):
    for a in values:
        give(a)


def test_average():
    values = [1, 2, 10, 20]

    with given() as gv:
        gv = gv.pipe(op.getitem("a"))

        results1 = []
        gv.pipe(op.average()).subscribe(results1.append)

        results2 = []
        gv.pipe(op.average(scan=True)).subscribe(results2.append)

        results3 = []
        gv.pipe(op.average(roll=2)).subscribe(results3.append)

        things(*values)

    assert results1 == [sum(values) / len(values)]
    assert results2 == [sum(values[:i]) / len(values[:i]) for i in range(1, len(values) + 1)]
    assert results3 == [values[0]] + [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]


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
        gv.pipe(op.count(lambda x: x > 0, roll=3)).subscribe(results4.append)

        things(*values)

    assert results1 == [len(values)]
    assert results2 == [len([v for v in values if v > 0])]
    assert results3 == [1, 2, 2, 3, 3, 3]
    assert results4 == [1, 2, 2, 2, 1, 1]


def test_affix():
    values = [1, 2, 3, 4]

    with given() as gv:
        results = []
        gv.pipe(
            op.affix(b=gv.pipe(op.getitem("a"), op.map(lambda x: x * x)))
        ).subscribe(results.append)

        things(*values)

    assert results == [{"a": x, "b": x * x} for x in values]


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

        varia()

    assert results1 == [d for d in everything if "x" in d]
    assert results2 == [d for d in everything if "y" in d and d["y"]]
    assert results3 == [d for d in everything if "x" in d and "y" in d and d["y"]]
    assert results4 == [d for d in everything if "x" in d and d["x"] > 10]
    assert results5 == [d for d in everything if "x" not in d]
