import builtins as __builtins

from rx.operators import (  # noqa: F401
    NotSet,
    all,
    amb,
    as_observable,
    buffer,
    buffer_toggle,
    buffer_when,
    buffer_with_count,
    buffer_with_time,
    buffer_with_time_or_count,
    cast,
    catch,
    combine_latest,
    concat,
    contains,
    datetime,
    debounce,
    default_if_empty,
    delay,
    delay_subscription,
    delay_with_mapper,
    dematerialize,
    distinct,
    distinct_until_changed,
    do,
    do_action,
    do_while,
    element_at,
    element_at_or_default,
    exclusive,
    expand,
    filter,
    filter_indexed,
    finally_action,
    find,
    find_index,
    first,
    first_or_default,
    flat_map,
    flat_map_indexed,
    flat_map_latest,
    fork_join,
    group_by,
    group_by_until,
    group_join,
    ignore_elements,
    is_empty,
    join,
    last,
    last_or_default,
    map,
    map_indexed,
    materialize,
    max_by,
    merge,
    merge_all,
    min_by,
    multicast,
    observe_on,
    on_error_resume_next,
    overload,
    pairwise,
    partition,
    partition_indexed,
    pipe,
    pluck,
    pluck_attr,
    publish,
    publish_value,
    reduce,
    ref_count,
    repeat,
    replay,
    retry,
    sample,
    scan,
    sequence_equal,
    share,
    single,
    single_or_default,
    single_or_default_async,
    skip,
    skip_last,
    skip_last_with_time,
    skip_until,
    skip_until_with_time,
    skip_while,
    skip_while_indexed,
    skip_with_time,
    slice,
    some,
    starmap,
    starmap_indexed,
    start_with,
    subscribe_on,
    switch_latest,
    take,
    take_last,
    take_last_buffer,
    take_last_with_time,
    take_until,
    take_until_with_time,
    take_while,
    take_while_indexed,
    take_with_time,
    throttle_first,
    throttle_with_mapper,
    throttle_with_timeout,
    time_interval,
    timedelta,
    timeout,
    timeout_with_mapper,
    timestamp,
    to_dict,
    to_future,
    to_iterable,
    to_list,
    to_marbles,
    to_set,
    typing,
    while_do,
    window,
    window_toggle,
    window_when,
    window_with_count,
    window_with_time,
    window_with_time_or_count,
    with_latest_from,
    zip,
    zip_with_iterable,
    zip_with_list,
)

from .utils import keyword_decorator

# Shortcut to throttle_first
throttle = throttle_first


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

    return map(_fmt)


def getitem(*names, strict=False):
    """Extract a key from a dictionary.

    Arguments:
        name: Name of the key to index with.
        strict: If true, every element in the stream is required to
            contains this key.
    """
    import operator

    if len(names) == 1:
        (name,) = names
        if strict:
            return map(operator.itemgetter(name))
        else:
            return pipe(filter(lambda arg: name in arg), map(operator.itemgetter(name)))
    else:
        if strict:
            return map(lambda arg: tuple(arg[name] for name in names))
        else:
            return pipe(
                filter(lambda arg: __builtins.all(name in arg for name in names)),
                map(lambda arg: tuple(arg[name] for name in names)),
            )


def keymap(fn):
    """Map a dict, passing keyword arguments.

    Arguments:
        fn: A function that will be called for each element, passing the
            element using **kwargs.

            Note: If the dict has elements that are not in the function's
            arguments list and the function does not have a **kwargs
            argument, these elements will be dropped and no error will
            occur.
    """
    import types

    if isinstance(fn, types.FunctionType):
        KWVAR_FLAG = 8
        co = fn.__code__
        if not co.co_flags & KWVAR_FLAG:
            fn = types.FunctionType(
                name=fn.__name__,
                code=co.replace(
                    co_flags=co.co_flags | KWVAR_FLAG,
                    # Add a dummy keyword argument with an illegal name
                    co_varnames=(*co.co_varnames, "#"),
                ),
                globals=fn.__globals__,
            )

    return map(lambda kwargs: fn(**kwargs))


def roll(n, reduce=None, seed=NotSet):  # noqa: F811
    """Group the last n elements, giving a sequence of overlapping sequences.

    This can be used to compute a rolling average of the 100 last element:
        op.roll(100, lambda xs: sum(xs) / len(xs))

    Arguments:
        n: The number of elements to group together.
        reduce: A function to reduce the group.

            It should take four arguments:
                last: The last result.
                add: The element that was just added. It is the last element
                    in the elements list.
                drop: The element that was dropped to make room for the
                    added one. It is *not* in the elements argument.
                    If the list of elements is not yet of size n, there is
                    no need to drop anything and drop is None.
                last_size: The window size on the last invocation.
                current_size: The window size on this invocation.

            Defaults to returning the deque of elements directly. The same
            reference is returned each time in order to save memory, so it
            should be processed immediately.
        seed: The first element of the reduction.
    """

    from collections import deque

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

        scan_command = scan(queue, seed)

    else:

        def queue(q, x):
            q.append(x)
            return q

        scan_command = scan(queue, q)

    return pipe(scan_command, stream_once())


class _Reducer:
    def __init__(self, reduce, roll):
        self.reduce = reduce
        self.roll = roll


@keyword_decorator
def reducer(func, default_seed=NotSet, postprocess=NotSet):
    op_scan = scan
    name = func.__name__
    if isinstance(func, type):
        constructor = func

    else:
        constructor = lambda: _Reducer(reduce=func, roll=None)

    def _create(*args, scan=False, seed=NotSet, **kwargs):
        reducer = constructor(*args, **kwargs)

        if seed is NotSet:
            seed = default_seed

        if scan is True:
            oper = op_scan(reducer.reduce, seed=seed)

        elif scan:
            oper = roll(n=scan, reduce=reducer.roll, seed=seed)

        else:
            oper = reduce(reducer.reduce, seed=seed)

        if postprocess is not NotSet:
            oper = pipe(oper, postprocess)

        return oper

    _create.__name__ = name
    return _create


@reducer(default_seed=(0, 0), postprocess=starmap(lambda x, sz: x / sz))
class average:
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
    postprocess=starmap(_average_and_variance_postprocess),
)
class average_and_variance:
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


def variance(*args, **kwargs):
    return pipe(average_and_variance(*args, **kwargs), starmap(lambda avg, var: var))


@reducer
def min(last, new):
    return __builtins.min(last, new)


@reducer
def max(last, new):
    return __builtins.max(last, new)


@reducer
def sum(last, new):
    return last + new


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


def affix(**streams):
    """Augment a stream of dicts with extra keys.

    The affixed streams should have the same length as the main one, so when
    affixing a reduction, one should set scan=True, or the roll option.

    Example:
        obs.affix(
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
        return {**main, **dict(__builtins.zip(keys, rest))}

    return pipe(zip(*values), map(merge))


def where(*keys, **conditions):
    """Filter entries with the given keys meeting the given conditions.

    Example:
        where("x", "!y", z=True, w=lambda x: x > 0)

    Arguments:
        keys: Keys that must be present in the dictionary or, if a key starts
            with "!", it must *not* be present.
        conditions: Maps a key to the value it must be associated to in the
            dictionary, or to a predicate function on the value.
    """
    from types import FunctionType

    conditions = {
        k: cond if isinstance(cond, FunctionType) else lambda x: cond == x
        for k, cond in conditions.items()
    }
    excluded = [k.lstrip("!") for k in keys if k.startswith("!")]
    keys = [k for k in keys if not k.startswith("!")]
    keys = (*keys, *conditions.keys())

    def filt(x):
        return (
            isinstance(x, dict)
            and __builtins.all(k in x for k in keys)
            and not __builtins.any(k in x for k in excluded)
            and __builtins.all(cond(x[k]) for k, cond in conditions.items())
        )

    return filter(filt)


def collect_between(start, end, common=None):
    """Collect all data between the start and end keys.

    Example:
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
    import rx

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


def stream_once():
    """Make sure that upstream operators only run once.

    Use this if upstream operators have side effects, otherwise each
    downstream subscription will re-run the effects.
    """

    import rx

    def go(source):
        observers = []

        def on_next(value):
            for obv in observers:
                obv.on_next(value)

        def on_error(value):
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


def tag(group="", field="$word", group_field="$group"):
    """Tag each dict or object with a unique word.

    If the item is a dict, do `item[field] = <new_word>`, otherwise
    attempt to do `setattr(item, field, <new_word>)`.

    These tags are displayed specially by the `display` method and they
    can be used to determine breakpoints with the `breakword` method.

    Arguments:
        group: An arbitrary group name that corresponds to an independent
            sequence of words. It determines the color in display.
        field: The field name in which to put the word
            (default: `$word`).
        group_field: The field name in which to put the group
            (default: `$group`).
    """
    try:
        import breakword as bw
    except ImportError:
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

    return pipe(map(tag_data), stream_once())


def unique():
    """Collect unique elements.

    Be aware that this keeps a set of all the elements seen so far,
    so it may prevent them from being reclaimed by garbage collection
    and can be expensive in memory.
    """
    import rx

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


def as_(key):
    """Make a stream of dictionaries using the given key.

    For example, [1, 2].as_("x") => [{"x": 1}, {"x": 2}]
    """
    return map(lambda x: {key: x})
