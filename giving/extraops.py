"""Extra operators for giving."""

import builtins
import inspect
import operator
from bisect import bisect_left
from collections import deque
from types import FunctionType

import reactivex as rx
from reactivex import operators as rxop
from reactivex.operators import NotSet

from .utils import lax_function, reducer


def _keyfn(key):
    if isinstance(key, (int, str)):
        return operator.itemgetter(key)
    elif key is None:
        return lambda x: x
    else:
        return key


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

                * **last**: The last result.
                * **add**: The element that was just added. It is the last element
                  in the elements list.
                * **drop**:
                  The element that was dropped to make room for the
                  added one. It is *not* in the elements argument.
                  If the list of elements is not yet of size n, there is
                  no need to drop anything and drop is None.
                * **last_size**: The window size on the last invocation.
                * **current_size**: The window size on this invocation.

            Defaults to returning the deque of elements directly.

            .. note::
                The same reference is returned each time in order to save memory, so it
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

    return rxop.compose(scan_command, rxop.share())


def affix(**streams):
    """Affix streams as extra keys on an existing stream of dicts.

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

        Or:

        .. code-block:: python

            obs.where("x", "y").affix(
                # o is obs.where("x", "y")
                minx=lambda o: o["x"].min(scan=True),
                xpy=lambda o: o["x", "y"].starmap(lambda x, y: x + y),
            )

    Arguments:
        streams: A mapping from extra keys to add to the dicts to Observables
            that generate the values, or to functions of one argument that will
            be called with the main Observable.
    """
    from .gvn import ObservableProxy

    keys = list(streams.keys())
    values = list(streams.values())

    def create(source):
        newobs = [
            v if hasattr(v, "subscribe") else v(ObservableProxy(source)) for v in values
        ]

        def merge(elems):
            main, *rest = elems
            return {**main, **dict(zip(keys, rest))}

        return rxop.compose(rxop.zip(*newobs), rxop.map(merge))(source)

    return create


def augment(**fns):
    """Augment a stream of dicts with new keys.

    Each key in ``fns`` should be associated to a function that will be called
    with the rest of the data as keyword arguments, so the argument names
    matter. The results overwrite the old data, if any keys are in common.

    .. note::
        The functions passed in ``fns`` will be wrapped with
        :func:`~giving.utils.lax_function` if possible.

        This means that these functions are considered to have an
        implicit ``**kwargs`` argument, so that any data they do
        not need is ignored.

    .. code-block:: python

        # [{"x": 1, "y": 2}, ...] => [{"x": 1, "y": 2, "z": 3}, ...]
        gv.augment(z=lambda x, y: x + y)

        # [{"lo": 2, "hi": 3}, ...] => [{"lo": 2, "hi": 3, "higher": 9}, ...]
        gv.augment(higher=lambda hi: hi * hi)

    Arguments:
        fns: A map from new key names to the functions to compute them.
    """

    fns = {k: lax_function(fn) for k, fn in fns.items()}
    return rxop.map(lambda data: {**{k: fn(**data) for k, fn in fns.items()}, **data})


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


def bottom(n=10, key=None, reverse=False):
    """Return the bottom n values, sorted in ascending order.

    .. marble::
        :alt: bottom

        ---1-2-7-3-9-0-|
        [    bottom(n=2)   ]
        ---------------0-1-|

    ``bottom`` may emit less than ``n`` elements, if there are
    less than ``n`` elements in the orginal sequence.

    Arguments:
        n: The number of bottom entries to return.
        key: The comparison key function to use or a string.
    """
    key = _keyfn(key)
    assert n > 0

    def update(entries, new):
        if entries is None:
            entries = ([], [])

        keyed, elems = entries
        newkey = key(new) if key else new

        if len(keyed) < n or (newkey > keyed[0] if reverse else newkey < keyed[-1]):
            ins = bisect_left(keyed, newkey)
            keyed.insert(ins, newkey)
            if reverse:
                ins = len(elems) - ins
            elems.insert(ins, new)
            if len(keyed) > n:
                del keyed[0 if reverse else -1]
                elems.pop()

        return keyed, elems

    return rxop.compose(
        rxop.reduce(update, seed=None),
        rxop.map(lambda x: () if x is None else x[1]),
        rxop.flat_map(lambda x: x),
    )


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

            return source.subscribe(
                on_next, obs.on_error, obs.on_completed, scheduler=scheduler
            )

        return rx.create(subscribe)

    return aggro


@reducer(default_seed=0)
class count:
    """Count operator.

    Returns an observable sequence containing a value that represents how many elements in the specified
    observable sequence satisfy a condition if provided, else the count of items.

    Arguments:
        predicate: A function to test each element for a condition.
        scan: If True, generate a running count, if a number *n*, count the number of elements/matches
            in the last *n* elements.
    """

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


def flatten(mapper=None):
    """Flatten a sequence of sequences/Observables into a single sequence.

    Without an argument, this is equivalent to ``flat_map(lambda x: x)``.

    .. marble::
        :alt: getitem

        --x1,x2-x3,x4-|
        [  flatten()  ]
        --x1-x2-x3-x4-|

    Arguments:
        mapper: A function applied to each element of the original sequence
            which should return a sequence to insert.
    """
    if mapper is None:

        def identity(x):
            return x

        mapper = identity
    return rxop.flat_map(mapper)


def format(string, raw=False, skip_missing=False):
    """Format an object using a format string.

    * If the data is a dict, it is passed as ``*kwargs`` to ``str.format``, unless raw=True
    * If the data is a tuple, it is passed as ``*args`` to ``str.format``, unless raw=True

    Arguments:
        string: The format string.
        raw: Whether to pass the data as ``*args`` or ``**kwargs`` if it is a tuple or dict.
        skip_missing: Whether to ignore KeyErrors due to missing entries in the format.
    """
    SKIP = object()

    def _fmt(x):
        if not raw and isinstance(x, dict):
            if skip_missing:
                try:
                    return string.format(**x)
                except KeyError:
                    return SKIP
            else:
                return string.format(**x)
        elif not raw and isinstance(x, tuple):
            return string.format(*x)
        else:
            return string.format(x)

    if skip_missing:
        return rxop.compose(
            rxop.map(_fmt),
            rxop.filter(lambda x: x is not SKIP),
        )
    else:
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
            return rxop.compose(
                rxop.filter(lambda arg: key in arg),
                rxop.map(operator.itemgetter(key)),
            )
    else:
        if strict:
            return rxop.map(lambda arg: tuple(arg[key] for key in keys))
        else:
            return rxop.compose(
                rxop.filter(lambda arg: builtins.all(key in arg for key in keys)),
                rxop.map(lambda arg: tuple(arg[key] for key in keys)),
            )


def group_wrap(name, **conditions):
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
        name: Name of the wrap block to group on.
        conditions: Maps a key to the value it must be associated to in the
            dictionary of the wrap statement, or to a predicate function on the
            value.
    """
    from .gvn import ObservableProxy

    def oper(source):
        opens = rxop.compose(
            where("$wrap", **conditions),
            getitem("$wrap"),
            where(step="begin", name=name),
        )

        def closes(data):
            return rxop.compose(
                getitem("$wrap", strict=False), where(step="end", id=data["id"])
            )(source)

        return source.pipe(
            where("!$wrap"),
            rxop.window_toggle(
                openings=opens(source),
                closing_mapper=closes,
            ),
            rxop.map(ObservableProxy),
        )

    return oper


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
        return {k2: data[k1] for k1, k2 in remap.items() if k1 in data}

    return rxop.compose(rxop.filter(_filt), rxop.map(_rekey))


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


def kmap(_fn=None, **_fns):
    """Map a dict, passing keyword arguments.

    ``kmap`` either takes a positional function argument or keyword arguments
    serving to build a new dict.

    Example:

        .. code-block:: python

            # [{"x": 1, "y": 2}] => [3]
            gv.kmap(lambda x, y: x + y)

            # [{"x": 1, "y": 2}] => [{"z": 3}]
            gv.kmap(z=lambda x, y: x + y)

    Arguments:
        _fn: A function that will be called for each element, passing the
            element using ``**kwargs``.

            .. note::

                If the dict has elements that are not in the function's
                arguments list and the function does not have a ``**kwargs``
                argument, these elements will be dropped and no error will
                occur.
        _fns: Alternatively, build a new dict with each key associated to
            a function with the same interface as fn.
    """
    if _fn and _fns or not _fn and not _fns:
        raise TypeError(
            "kmap either takes one argument or keyword arguments but not both"
        )

    elif _fn:
        _fn = lax_function(_fn)
        return rxop.map(lambda data: _fn(**data))

    else:
        fns = {k: lax_function(fn) for k, fn in _fns.items()}
        return rxop.map(lambda data: {k: fn(**data) for k, fn in fns.items()})


def kmerge(scan=False):
    """Merge the dictionaries in the stream.

    .. marble::
        :alt: kmerge

        --x1--y2--x3--z4-|
        [          kmerge()        ]
        -----------------x3,y2,z4--|

    .. marble::
        :alt: kmerge2

        --x1--y2-----x3-----z4-------|
        [      kmerge(scan=True)     ]
        --x1--x1,y2--x3,y2--x3,y2,z4-|
    """

    def _merge(last, new):
        return {**last, **new}

    if scan:
        assert isinstance(scan, bool)
        return rxop.scan(_merge)
    else:
        return rxop.reduce(_merge)


def kscan():
    """Alias for ``kmerge(scan=True)``."""
    return kmerge(scan=True)


@reducer
class min:
    """Produce the minimum of a stream of values.

    .. marble::
        :alt: minimum

        --3--2--7--6-|
        [     min()    ]
        -------------2-|

    Arguments:
        key: A key mapping function or a string.
        comparer: A function of two elements that returns -1 if the first is smaller
            than the second, 0 if they are equal, 1 if the second is larger.
        scan: If True, generate the current minimum on every element.
        seed: First element of the reduction.
    """

    def __init__(self, key=None, comparer=None):
        self.comparer = comparer or operator.gt
        self.key = _keyfn(key)

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
        key: A key mapping function or a string.
        comparer: A function of two elements that returns -1 if the first is smaller
            than the second, 0 if they are equal, 1 if the second is larger.
        scan: If True, generate the current maximum on every element.
        seed: First element of the reduction.
    """

    def __init__(self, key=None, comparer=None):
        self.comparer = comparer or operator.gt
        self.key = _keyfn(key)

    def reduce(self, last, new):
        lastc = self.key(last)
        newc = self.key(new)
        if self.comparer(lastc, newc) > 0:
            return last
        else:
            return new


def sole(*, keep_key=False, exclude=[]):
    """Extract values from a stream of dicts with one entry each.

    .. marble::
        :alt: sole

        --x1--y2--y3--z4--|
        [     sole()      ]
        --1---2---3---4---|

    If, after removing keys from the exclusion set, any dict is empty
    or has a length superior to 1, that is an error.

    Arguments:
        keep_key: If True, return a (key, value) tuple, otherwise only
            return the value. Defaults to False.
        exclude: Keys to exclude.
    """

    if isinstance(exclude, str):
        exclude = [exclude]

    def extract(data):
        pairs = [(k, v) for k, v in data.items() if k not in exclude]
        if len(pairs) == 1:
            (kv,) = pairs
            return kv if keep_key else kv[1]
        elif len(pairs) == 0:
            raise Exception("The dict has no valid entries", data)
        else:
            raise Exception("The dict has more than one entry", data)

    return rxop.map(extract)


def sort(key=None, reverse=False):
    """Sort the stream.

    .. marble::
        :alt: bottom

        ---1-2-7-3-9-0-|
        [    sort()    ]
        ---------------0-1-2-3-7-9-|

    Arguments:
        key: The comparison key function to use or a string.
        reverse: If True, the sort is descending.
    """
    key = _keyfn(key)

    return rxop.compose(
        rxop.to_list(),
        rxop.map(lambda xs: list(sorted(xs, key=key, reverse=reverse))),
        rxop.flat_map(lambda x: x),
    )


@reducer
def sum(last, new):
    return last + new


def tag(group="", field="$word", group_field="$group"):
    """Tag each dict or object with a unique word.

    If the item is a dict, do ``item[field] = <new_word>``, otherwise
    attempt to do ``setattr(item, field, <new_word>)``.

    These tags are displayed specially by the
    :meth:`~giving.gvn.ObservableProxy.display` method and they
    can be used to determine breakpoints with the
    :meth:`~giving.gvn.ObservableProxy.breakword` method.

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

    return rxop.compose(rxop.map(tag_data), rxop.share())


def top(n=10, key=None):
    """Return the top n values, sorted in descending order.

    .. marble::
        :alt: top

        ---1-2-7-3-9-0-|
        [     top(n=2)     ]
        ---------------9-7-|

    ``top`` may emit less than ``n`` elements, if there are
    less than ``n`` elements in the orginal sequence.

    Arguments:
        n: The number of top entries to return.
        key: The comparison key function to use or a string.
    """
    return bottom(n=n, key=key, reverse=True)


def variance(*args, **kwargs):
    return rxop.compose(
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


def wmap(name, fn=None, pass_keys=True):
    """Map each begin/end pair of a give.wrap.

    In this schema, B and E correspond to the messages sent in the enter and exit
    phases respectively of the :meth:`~Giver.wrap` context manager.

    .. marble::
        :alt: group_wrap

        --B1-B2-E2-B3-E3-E1--|
        [        wmap()      ]
        --------x2----x3-x1--|

    Example:
        .. code-block:: python

            def _wrap(x):
                yield
                return x * 10

            with given() as gv:
                results = gv.wmap("block", _wrap).accum()

                with give.wrap("block", x=3):
                    with give.wrap("block", x=4):
                        pass

            assert results == [40, 30]

    Arguments:
        name: Name of the wrap block to group on.
        fn: A generator function that yields exactly once.
        pass_keys: Whether to pass the arguments to give.wrap() as
            keyword arguments at the start (defaults to True).
    """
    if fn is None and not isinstance(name, str):
        name, fn = None, name

    if not inspect.isgeneratorfunction(fn):
        raise TypeError("wmap() must take a generator function")

    if pass_keys:
        fn = lax_function(fn)

    def aggro(source):
        def subscribe(obs, scheduler=None):
            managers = {}

            def on_next(data):
                wr = data.get("$wrap", None)
                if wr is None or (name is not None and wr["name"] != name):
                    return

                if wr["step"] == "begin":
                    key = wr["id"]
                    assert key not in managers
                    if pass_keys:
                        manager = fn(**data)
                    else:
                        manager = fn()

                    managers[key] = manager
                    manager.send(None)

                if wr["step"] == "end":
                    key = wr["id"]
                    manager = managers[key]
                    try:
                        manager.send(data)
                    except StopIteration as stop:
                        obs.on_next(stop.value)
                    else:
                        raise Exception("Function in wmap() should yield exactly once.")
                    del managers[key]

            return source.subscribe(
                on_next, obs.on_error, obs.on_completed, scheduler=scheduler
            )

        return rx.create(subscribe)

    return aggro


__all__ = [
    "affix",
    "as_",
    "augment",
    "average",
    "average_and_variance",
    "bottom",
    "collect_between",
    "count",
    "flatten",
    "format",
    "getitem",
    "group_wrap",
    "keep",
    "kfilter",
    "kmap",
    "kmerge",
    "kscan",
    "max",
    "min",
    "roll",
    "sole",
    "sort",
    "sum",
    "tag",
    "top",
    "variance",
    "where",
    "where_any",
    "wmap",
]
