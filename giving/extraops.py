import builtins
import operator
from collections import deque
from types import FunctionType

import rx
from rx import operators as rxop
from rx.operators import NotSet

from .utils import keyword_decorator, lax_function

###########
# Reducer #
###########


class _Reducer:
    def __init__(self, reduce, roll):
        self.reduce = reduce
        self.roll = roll


@keyword_decorator
def reducer(func, default_seed=NotSet, postprocess=NotSet):
    name = func.__name__
    if isinstance(func, type):
        constructor = func

    else:

        def constructor():
            return _Reducer(reduce=func, roll=None)

    def _create(*args, scan=False, seed=NotSet, **kwargs):
        reducer = constructor(*args, **kwargs)

        if seed is NotSet:
            seed = default_seed

        if scan is True:
            oper = rxop.scan(reducer.reduce, seed=seed)

        elif scan:
            oper = roll(n=scan, reduce=reducer.roll, seed=seed)

        else:
            oper = rxop.reduce(reducer.reduce, seed=seed)

        if postprocess is not NotSet:
            oper = rxop.pipe(oper, postprocess)

        return oper

    _create.__name__ = name
    _create.__doc__ = func.__doc__
    return _create


##############################
# Extra operators for giving #
##############################


def affix(**streams):
    """Augment a stream of dicts with extra keys.

    The affixed streams should have the same length as the main one, so when
    affixing a reduction, one should set ``scan=True``, or ``scan=n``.

    .. marble::
        :alt: affix

         ---a1---------a2-----------a3-------|
        x----4-----------5---------6---------|
        y-----7--------8----------9----------|
        [           affix(b=x, c=y)          ]
         -----a1,b4,c7---a2,b5,c8---a3,b6,c9-|

    Example:
        .. code-block:: python

            obs.where("x", "y").affix(
                minx=obs["x"].min(scan=True),
                xpy=obs["x", "y"].starmap(lambda x, y: x + y),
            )

    Arguments:
        streams: A mapping from extra keys to add to the dicts to Observables
            that generate the values.
    """

    keys = list(streams.keys())
    values = list(streams.values())

    def merge(elems):
        main, *rest = elems
        return {**main, **dict(zip(keys, rest))}

    return rxop.pipe(rxop.zip(*values), rxop.map(merge))


def as_(key):
    """Make a stream of dictionaries using the given key.

    For example, ``[1, 2].as_("x")`` => ``[{"x": 1}, {"x": 2}]``

    .. marble::
        :alt: as_

        --1---2---3--|
        [   as_(x)   ]
        --x1--x2--x3-|

    Arguments:
        key: Key under which to generate each element of the stream.
    """
    return rxop.map(lambda x: {key: x})


@reducer(default_seed=(0, 0), postprocess=rxop.starmap(lambda x, sz: x / sz))
class average:
    """Produce the average of a stream of values.

    .. marble::
        :alt: average

        --1--3--5-|
        [ average() ]
        ----------3-|

    .. marble::
        :alt: average2

        ----1----3----5------|
        [ average(scan=True) ]
        ----1----2----3------|

    .. marble::
        :alt: average3

        ----1----3----5---|
        [ average(scan=2) ]
        ----1----2----4---|

    Arguments:
        scan: If True, generate the current average on every element. If a number *n*,
            generate the average on the last *n* elements.
        seed: First element of the reduction.
    """

    def reduce(self, last, add):
        x, sz = last
        return (x + add, sz + 1)

    def roll(self, last, add, drop, last_size, current_size):
        x, _ = last
        if last_size == current_size:
            return (x + add - drop, current_size)
        else:
            return (x + add, current_size)


def _average_and_variance_postprocess(sm, v2, sz):
    avg = sm / sz
    if sz >= 2:
        var = v2 / (sz - 1)
    else:
        var = None
    return (avg, var)


@reducer(
    default_seed=(0, 0, 0),
    postprocess=rxop.starmap(_average_and_variance_postprocess),
)
class average_and_variance:
    """Produce the average and variance of a stream of values.

    .. note::

        The variance for the first element is always None.

    Arguments:
        scan: If True, generate the current average+variance on every element.
            If a number *n*, generate the average+variance on the last *n* elements.
    """

    def reduce(self, last, add):
        prev_sum, prev_v2, prev_size = last
        new_size = prev_size + 1
        new_sum = prev_sum + add
        if prev_size:
            prev_mean = prev_sum / prev_size
            new_mean = new_sum / new_size
            new_v2 = prev_v2 + (add - prev_mean) * (add - new_mean)
        else:
            new_v2 = prev_v2
        return (new_sum, new_v2, new_size)

    def roll(self, last, add, drop, last_size, current_size):
        if last_size == current_size:
            prev_sum, prev_v2, prev_size = last
            new_sum = prev_sum - drop + add
            prev_mean = prev_sum / prev_size
            new_mean = new_sum / prev_size
            new_v2 = (
                prev_v2
                + (add - prev_mean) * (add - new_mean)
                - (drop - prev_mean) * (drop - new_mean)
            )
            return (new_sum, new_v2, prev_size)
        else:
            return self.reduce(last, add)


def collect_between(start, end, common=None):
    """Collect all data between the start and end keys.

    Example:
        .. code-block:: python

            with given() as gv:
                gv.collect_between("A", "Z") >> (results := [])
                give(A=1)
                give(B=2)
                give(C=3, D=4, A=5)
                give(Z=6)
                assert results == [{"A": 5, "B": 2, "C": 3, "D": 4, "Z": 6}]

    Arguments:
        start: The key that marks the beginning of the accumulation.
        end: The key that marks the end of the accumulation.
        common: A key that must be present in all data and must have
            the same value in the whole group.
    """

    def aggro(source):
        def subscribe(obs, scheduler=None):
            dicts = {}

            def on_next(value):
                if isinstance(value, dict):
                    if common is not None:
                        if common in value:
                            key = value[common]
                        else:
                            return
                    else:
                        key = None
                    if start in value:
                        dicts[key] = dict(value)
                    elif end in value:
                        current = dicts.setdefault(key, {})
                        current.update(value)
                        obs.on_next(current)
                        del dicts[key]
                    else:
                        current = dicts.setdefault(key, {})
                        current.update(value)

            return source.subscribe(on_next, obs.on_error, obs.on_completed, scheduler)

        return rx.create(subscribe)

    return aggro


@reducer(default_seed=0)
class count:
    def __init__(self, predicate=None):
        self.predicate = predicate

    def reduce(self, last, new):
        if self.predicate is None or self.predicate(new):
            return last + 1
        else:
            return last

    def roll(self, last, new, drop, last_size, current_size):
        if last_size == current_size:
            if self.predicate is None:
                return last
            else:
                plus = 1 if self.predicate(new) else 0
                minus = 1 if self.predicate(drop) else 0
                return last + plus - minus
        else:
            return self.reduce(last, new)


def format(string):
    """Format an object using a format string.

    Arguments:
        string: The format string.
    """

    def _fmt(x):
        if isinstance(x, dict):
            return string.format(**x)
        elif isinstance(x, (list, tuple)):
            return string.format(*x)
        else:
            return string.format(x)

    return rxop.map(_fmt)


def getitem(*keys, strict=False):
    """Extract one or more keys from a dictionary.

    If more than one key is given, a stream of tuples is produced.

    .. marble::
        :alt: getitem

        --x1--x2--x3-|
        [ getitem(x) ]
        --1---2---3--|

    Arguments:
        keys: Names of the keys to index with.
        strict: If true, every element in the stream is required to
            contains this key.
    """
    if len(keys) == 1:
        (key,) = keys
        if strict:
            return rxop.map(operator.itemgetter(key))
        else:
            return rxop.pipe(
                rxop.filter(lambda arg: key in arg),
                rxop.map(operator.itemgetter(key)),
            )
    else:
        if strict:
            return rxop.map(lambda arg: tuple(arg[key] for key in keys))
        else:
            return rxop.pipe(
                rxop.filter(lambda arg: builtins.all(key in arg for key in keys)),
                rxop.map(lambda arg: tuple(arg[key] for key in keys)),
            )


def group_wrap(*keys, **conditions):
    """Return a stream of observables for wrapped groups.

    In this schema, B and E correspond to the messages sent in the enter and exit
    phases respectively of the :meth:`~Giver.wrap` context manager.

    .. marble::
        :alt: group_wrap

        --B-a1-a2-E-a3-B-a4-E--|
        [     group_wrap()     ]
        ---+-----------+-------|
           +a1-a2-|
                       +-a4-|

    Example:
        .. code-block:: python

            results = []

            @obs.group_wrap().subscribe
            def _(obs2):
                obs2["a"].sum() >> results

    Arguments:
        keys: Keys that must be present in the dictionary of the wrap statement
            or, if a key starts with "!", it must *not* be present.
        conditions: Maps a key to the value it must be associated to in the
            dictionary of the wrap statement, or to a predicate function on the
            value.
    """
    from .obs import ObservableProxy

    begin = "$begin"
    end = "$end"

    def oper(source):
        return source.pipe(
            where(f"!{begin}", f"!{end}"),
            rxop.window_toggle(
                openings=where(begin, *keys, **conditions)(source),
                closing_mapper=lambda data: where(**{end: data[begin]})(source),
            ),
            rxop.map(ObservableProxy),
        )

    return oper


def kcombine():
    """Incrementally merge the dictionaries in the stream.

    .. marble::
        :alt: kcombine

        --x1--y2-----x3-----z4-------|
        [         kcombine()         ]
        --x1--x1,y2--x3,y2--x3,y2,z4-|
    """

    def _combine(last, new):
        return {**last, **new}

    return rxop.scan(_combine)


def keep(*keys, **remap):
    """Keep certain dict keys and remap others.

    .. marble::
        :alt: keep

        --x1,z2--y3--z4--|
        [   keep(x, y)   ]
        --x1-----y3------|

    Arguments:
        keys: Keys that must be kept
        remap: Keys that must be renamed
    """
    remap = {**{k: k for k in keys}, **remap}

    def _filt(data):
        return isinstance(data, dict) and any(k in remap for k in data.keys())

    def _rekey(data):
        return {remap[k]: v for k, v in data.items() if k in remap}

    return rxop.pipe(rxop.filter(_filt), rxop.map(_rekey))


def kfilter(fn):
    """Filter a stream of dictionaries.

    Example:

        .. code-block:: python

            # [{"x": 1, "y": 2}, {"x": 100, "y": 50}] => [{"x": 100, "y": 50}]
            gv.kfilter(lambda x, y: x > y)

    Arguments:
        fn: A function that will be called for each element, passing the
            element using ``**kwargs``.

            .. note::

                If the dict has elements that are not in the function's
                arguments list and the function does not have a ``**kwargs``
                argument, these elements will be dropped and no error will
                occur.
    """
    fn = lax_function(fn)
    return rxop.filter(lambda kwargs: fn(**kwargs))


def kmap(_fn=None, **_kwargs):
    """Map a dict, passing keyword arguments.

    ``kmap`` either takes a positional function argument or keyword arguments
    serving to build a new dict.

    Example:

        .. code-block:: python

            # [{"x": 1, "y": 2}] => [3]
            gv.kmap(lambda x, y: x + y)

            # [{"x": 1, "y": 2}] => [{"x": 1, "y": 2, "z": 3}]
            gv.kmap(z=lambda x, y: x + y)

    Arguments:
        _fn: A function that will be called for each element, passing the
            element using ``**kwargs``.

            .. note::

                If the dict has elements that are not in the function's
                arguments list and the function does not have a ``**kwargs``
                argument, these elements will be dropped and no error will
                occur.
        _kwargs: Alternatively, build a new dict with each key associated to
            a function with the same interface as fn.
    """
    if _fn and _kwargs or not _fn and not _kwargs:
        raise TypeError(
            "kmap either takes one argument or keyword arguments but not both"
        )

    elif _fn:
        _fn = lax_function(_fn)
        return rxop.map(lambda kwargs: _fn(**kwargs))

    else:
        fns = {k: lax_function(fn) for k, fn in _kwargs.items()}
        return rxop.map(lambda kwargs: {k: fn(**kwargs) for k, fn in fns.items()})


@reducer
class min:
    """Produce the minimum of a stream of values.

    .. marble::
        :alt: minimum

        --3--2--7--6-|
        [     min()    ]
        -------------2-|

    Arguments:
        key: A key mapping function.
        comparer: A function of two elements that returns -1 if the first is smaller
            than the second, 0 if they are equal, 1 if the second is larger.
        scan: If True, generate the current minimum on every element.
        seed: First element of the reduction.
    """

    def __init__(self, key=None, comparer=None):
        self.comparer = comparer or operator.gt
        self.key = key or (lambda x: x)

    def reduce(self, last, new):
        lastc = self.key(last)
        newc = self.key(new)
        if self.comparer(lastc, newc) <= 0:
            return last
        else:
            return new


@reducer
class max:
    """Produce the maximum of a stream of values.

    .. marble::
        :alt: maximum

        --3--2--7--6-|
        [     max()    ]
        -------------7-|

    Arguments:
        key: A key mapping function.
        comparer: A function of two elements that returns -1 if the first is smaller
            than the second, 0 if they are equal, 1 if the second is larger.
        scan: If True, generate the current maximum on every element.
        seed: First element of the reduction.
    """

    def __init__(self, key=None, comparer=None):
        self.comparer = comparer or operator.gt
        self.key = key or (lambda x: x)

    def reduce(self, last, new):
        lastc = self.key(last)
        newc = self.key(new)
        if self.comparer(lastc, newc) > 0:
            return last
        else:
            return new


def roll(n, reduce=None, seed=NotSet):  # noqa: F811
    """Group the last n elements, giving a sequence of overlapping sequences.

    For example, this can be used to compute a rolling average of the 100 last
    elements (however, ``average(scan=100)`` is better optimized).

    .. code-block:: python

        op.roll(100, lambda xs: sum(xs) / len(xs))

    .. marble::
        :alt: roll

        --1--2--3---4---5---6---|
        [         roll(3)       ]
        --1--12-123-234-345-456-|

    Arguments:
        n: The number of elements to group together.
        reduce: A function to reduce the group.

            It should take five arguments:
                * last: The last result.
                * add: The element that was just added. It is the last element
                    in the elements list.
                * drop: The element that was dropped to make room for the
                    added one. It is *not* in the elements argument.
                    If the list of elements is not yet of size n, there is
                    no need to drop anything and drop is None.
                * last_size: The window size on the last invocation.
                * current_size: The window size on this invocation.

            Defaults to returning the deque of elements directly. The same
            reference is returned each time in order to save memory, so it
            should be processed immediately.
        seed: The first element of the reduction.
    """
    q = deque(maxlen=n)

    if reduce is not None:

        def queue(current, x):
            drop = q[0] if len(q) == n else NotSet
            last_size = len(q)
            q.append(x)
            current_size = len(q)
            return reduce(
                current,
                x,
                drop=drop,
                last_size=last_size,
                current_size=current_size,
            )

        scan_command = rxop.scan(queue, seed)

    else:

        def queue(q, x):
            q.append(x)
            return q

        scan_command = rxop.scan(queue, q)

    return rxop.pipe(scan_command, stream_once())


def stream_once():
    """Make sure that upstream operators only run once.

    Use this if upstream operators have side effects, otherwise each
    downstream subscription will re-run the effects.
    """

    def go(source):
        observers = []

        def on_next(value):
            for obv in observers:
                obv.on_next(value)

        def on_error(value):  # pragma: no cover
            for obv in observers:
                obv.on_error(value)

        def on_completed():
            for obv in observers:
                obv.on_completed()

        def subscribe(obv, scheduler):
            observers.append(obv)
            return dispo

        dispo = source.subscribe_(on_next, on_error, on_completed)
        return rx.Observable(subscribe)

    return go


@reducer
def sum(last, new):
    return last + new


def tag(group="", field="$word", group_field="$group"):
    """Tag each dict or object with a unique word.

    If the item is a dict, do ``item[field] = <new_word>``, otherwise
    attempt to do ``setattr(item, field, <new_word>)``.

    These tags are displayed specially by the
    :meth:`~giving.obs.ObservableProxy.display` method and they
    can be used to determine breakpoints with the
    :meth:`~giving.obs.ObservableProxy.breakword` method.

    Arguments:
        group: An arbitrary group name that corresponds to an independent
            sequence of words. It determines the color in display.
        field: The field name in which to put the word
            (default: ``$word``).
        group_field: The field name in which to put the group
            (default: ``$group``).
    """
    try:
        import breakword as bw
    except ImportError:  # pragma: no cover
        raise ImportError(
            "Package `breakword` must be installed to use the tag() operator"
        )

    grp = bw.groups[group]

    def tag_data(data):
        word = grp.gen()
        if isinstance(data, dict):
            data = {field: word, **data}
            if group:
                data[group_field] = group
        else:
            if group:
                setattr(data, group_field, group)
                setattr(data, "$group", group)
            setattr(data, field, word)
            setattr(data, "$word", word)
        return data

    return rxop.pipe(rxop.map(tag_data), stream_once())


def unique():
    """Collect unique elements.

    Be aware that this keeps a set of all the elements seen so far,
    so it may prevent them from being reclaimed by garbage collection
    and can be expensive in memory.

    .. marble::
        :alt: unique

        -1-3-1-2-2-5-|
        [  unique()  ]
        -1-3---2---5-|

    """

    def oper(source):
        def subscribe(obv, scheduler=None):
            elements = set()

            def on_next(value):
                if isinstance(value, dict):
                    key = tuple(value.items())
                else:
                    key = value

                if key not in elements:
                    elements.add(key)
                    obv.on_next(value)

            return source.subscribe(on_next, obv.on_error, obv.on_completed, scheduler)

        return rx.create(subscribe)

    return oper


def variance(*args, **kwargs):
    return rxop.pipe(
        average_and_variance(*args, **kwargs), rxop.starmap(lambda avg, var: var)
    )


def where(*keys, **conditions):
    """Filter entries with the given keys meeting the given conditions.

    .. marble::
        :alt: where

        ---a1--b2--c3--|
        [  where(b=2)  ]
        -------b2------|

    .. marble::
        :alt: where2

        ---a1--b2--a3--|
        [   where(a)   ]
        ---a1------a3--|

    Example:
        .. code-block:: python

            where("x", "!y", z=True, w=lambda x: x > 0)

    Arguments:
        keys: Keys that must be present in the dictionary or, if a key starts
            with "!", it must *not* be present.
        conditions: Maps a key to the value it must be associated to in the
            dictionary, or to a predicate function on the value.
    """
    conditions = {
        k: cond if isinstance(cond, FunctionType) else lambda x, value=cond: value == x
        for k, cond in conditions.items()
    }
    excluded = [k.lstrip("!") for k in keys if k.startswith("!")]
    keys = [k for k in keys if not k.startswith("!")]
    keys = (*keys, *conditions.keys())

    def filt(x):
        return (
            isinstance(x, dict)
            and builtins.all(k in x for k in keys)
            and not builtins.any(k in x for k in excluded)
            and builtins.all(cond(x[k]) for k, cond in conditions.items())
        )

    return rxop.filter(filt)


def where_any(*keys):
    """Filter entries with any of the given keys.

    .. marble::
        :alt: where_any

        ---a1--b2--c3--|
        [ where_any(b) ]
        -------b2------|

    Arguments:
        keys: Keys that must be present in the dictionary.
    """

    def _filt(data):
        return isinstance(data, dict) and any(k in data for k in keys)

    return rxop.filter(_filt)
