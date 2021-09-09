import inspect
from contextlib import contextmanager
from functools import wraps

from . import operators as op
from .executors import Breakpoint, Displayer
from .utils import lax_function


def _opmethod(operator):
    @wraps(operator)
    def f(self, *args, **kwargs):
        return self.pipe(operator(*args, **kwargs))

    return f


def prox(obs):
    assert not isinstance(obs, ObservableProxy)
    return ObservableProxy(obs)


class ObservableProxy:
    def __init__(self, obs):
        self.obs = obs

    def pipe(self, *args, **kwargs):
        return prox(self.obs.pipe(*args, **kwargs))

    def subscribe(self, *args, **kwargs):
        return self.obs.subscribe(*args, **kwargs)

    def subscribe_(self, *args, **kwargs):
        return self.obs.subscribe_(*args, **kwargs)

    def __rshift__(self, subscription):
        if isinstance(subscription, list):
            subscription = subscription.append
        elif isinstance(subscription, set):
            subscription = subscription.add
        return self.subscribe(subscription)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        strict = not any("?" in x for x in item if isinstance(x, str))
        item = [x.lstrip("?") if isinstance(x, str) else x for x in item]
        return self.getitem(*item, strict=strict)

    #######################
    # Native rx operators #
    #######################

    all = _opmethod(op.all)
    amb = _opmethod(op.amb)
    as_observable = _opmethod(op.as_observable)
    average = _opmethod(op.average)
    buffer = _opmethod(op.buffer)
    buffer_toggle = _opmethod(op.buffer_toggle)
    buffer_when = _opmethod(op.buffer_when)
    buffer_with_count = _opmethod(op.buffer_with_count)
    buffer_with_time = _opmethod(op.buffer_with_time)
    buffer_with_time_or_count = _opmethod(op.buffer_with_time_or_count)
    cast = _opmethod(op.cast)
    catch = _opmethod(op.catch)
    combine_latest = _opmethod(op.combine_latest)
    concat = _opmethod(op.concat)
    contains = _opmethod(op.contains)
    count = _opmethod(op.count)
    datetime = _opmethod(op.datetime)
    debounce = _opmethod(op.debounce)
    default_if_empty = _opmethod(op.default_if_empty)
    delay = _opmethod(op.delay)
    delay_subscription = _opmethod(op.delay_subscription)
    delay_with_mapper = _opmethod(op.delay_with_mapper)
    dematerialize = _opmethod(op.dematerialize)
    distinct = _opmethod(op.distinct)
    distinct_until_changed = _opmethod(op.distinct_until_changed)
    do = _opmethod(op.do)
    do_action = _opmethod(op.do_action)
    do_while = _opmethod(op.do_while)
    element_at = _opmethod(op.element_at)
    element_at_or_default = _opmethod(op.element_at_or_default)
    exclusive = _opmethod(op.exclusive)
    expand = _opmethod(op.expand)
    filter = _opmethod(op.filter)
    filter_indexed = _opmethod(op.filter_indexed)
    finally_action = _opmethod(op.finally_action)
    find = _opmethod(op.find)
    find_index = _opmethod(op.find_index)
    first = _opmethod(op.first)
    first_or_default = _opmethod(op.first_or_default)
    flat_map = _opmethod(op.flat_map)
    flat_map_indexed = _opmethod(op.flat_map_indexed)
    flat_map_latest = _opmethod(op.flat_map_latest)
    fork_join = _opmethod(op.fork_join)
    group_by = _opmethod(op.group_by)
    group_by_until = _opmethod(op.group_by_until)
    group_join = _opmethod(op.group_join)
    ignore_elements = _opmethod(op.ignore_elements)
    is_empty = _opmethod(op.is_empty)
    join = _opmethod(op.join)
    last = _opmethod(op.last)
    last_or_default = _opmethod(op.last_or_default)
    map = _opmethod(op.map)
    map_indexed = _opmethod(op.map_indexed)
    materialize = _opmethod(op.materialize)
    max = _opmethod(op.max)
    max_by = _opmethod(op.max_by)
    merge = _opmethod(op.merge)
    merge_all = _opmethod(op.merge_all)
    min = _opmethod(op.min)
    min_by = _opmethod(op.min_by)
    multicast = _opmethod(op.multicast)
    observe_on = _opmethod(op.observe_on)
    on_error_resume_next = _opmethod(op.on_error_resume_next)
    overload = _opmethod(op.overload)
    pairwise = _opmethod(op.pairwise)
    partition = _opmethod(op.partition)
    partition_indexed = _opmethod(op.partition_indexed)
    pluck = _opmethod(op.pluck)
    pluck_attr = _opmethod(op.pluck_attr)
    publish = _opmethod(op.publish)
    publish_value = _opmethod(op.publish_value)
    reduce = _opmethod(op.reduce)
    ref_count = _opmethod(op.ref_count)
    repeat = _opmethod(op.repeat)
    replay = _opmethod(op.replay)
    retry = _opmethod(op.retry)
    sample = _opmethod(op.sample)
    scan = _opmethod(op.scan)
    sequence_equal = _opmethod(op.sequence_equal)
    share = _opmethod(op.share)
    single = _opmethod(op.single)
    single_or_default = _opmethod(op.single_or_default)
    single_or_default_async = _opmethod(op.single_or_default_async)
    skip = _opmethod(op.skip)
    skip_last = _opmethod(op.skip_last)
    skip_last_with_time = _opmethod(op.skip_last_with_time)
    skip_until = _opmethod(op.skip_until)
    skip_until_with_time = _opmethod(op.skip_until_with_time)
    skip_while = _opmethod(op.skip_while)
    skip_while_indexed = _opmethod(op.skip_while_indexed)
    skip_with_time = _opmethod(op.skip_with_time)
    slice = _opmethod(op.slice)
    some = _opmethod(op.some)
    starmap = _opmethod(op.starmap)
    starmap_indexed = _opmethod(op.starmap_indexed)
    start_with = _opmethod(op.start_with)
    subscribe_on = _opmethod(op.subscribe_on)
    sum = _opmethod(op.sum)
    switch_latest = _opmethod(op.switch_latest)
    take = _opmethod(op.take)
    take_last = _opmethod(op.take_last)
    take_last_buffer = _opmethod(op.take_last_buffer)
    take_last_with_time = _opmethod(op.take_last_with_time)
    take_until = _opmethod(op.take_until)
    take_until_with_time = _opmethod(op.take_until_with_time)
    take_while = _opmethod(op.take_while)
    take_while_indexed = _opmethod(op.take_while_indexed)
    take_with_time = _opmethod(op.take_with_time)
    throttle_first = _opmethod(op.throttle_first)
    throttle_with_mapper = _opmethod(op.throttle_with_mapper)
    throttle_with_timeout = _opmethod(op.throttle_with_timeout)
    time_interval = _opmethod(op.time_interval)
    timedelta = _opmethod(op.timedelta)
    timeout = _opmethod(op.timeout)
    timeout_with_mapper = _opmethod(op.timeout_with_mapper)
    timestamp = _opmethod(op.timestamp)
    to_dict = _opmethod(op.to_dict)
    to_future = _opmethod(op.to_future)
    to_iterable = _opmethod(op.to_iterable)
    to_list = _opmethod(op.to_list)
    to_marbles = _opmethod(op.to_marbles)
    to_set = _opmethod(op.to_set)
    typing = _opmethod(op.typing)
    while_do = _opmethod(op.while_do)
    window = _opmethod(op.window)
    window_toggle = _opmethod(op.window_toggle)
    window_when = _opmethod(op.window_when)
    window_with_count = _opmethod(op.window_with_count)
    window_with_time = _opmethod(op.window_with_time)
    window_with_time_or_count = _opmethod(op.window_with_time_or_count)
    with_latest_from = _opmethod(op.with_latest_from)
    zip = _opmethod(op.zip)
    zip_with_iterable = _opmethod(op.zip_with_iterable)
    zip_with_list = _opmethod(op.zip_with_list)

    #############################
    # giving-specific operators #
    #############################

    affix = _opmethod(op.affix)
    as_ = _opmethod(op.as_)
    average_and_variance = _opmethod(op.average_and_variance)
    collect_between = _opmethod(op.collect_between)
    format = _opmethod(op.format)
    getitem = _opmethod(op.getitem)
    kcombine = _opmethod(op.kcombine)
    keep = _opmethod(op.keep)
    kfilter = _opmethod(op.kfilter)
    kmap = _opmethod(op.kmap)
    roll = _opmethod(op.roll)
    stream_once = _opmethod(op.stream_once)
    tag = _opmethod(op.tag)
    throttle = _opmethod(op.throttle)
    unique = _opmethod(op.unique)
    variance = _opmethod(op.variance)
    where = _opmethod(op.where)

    def breakpoint(self, **kwargs):  # pragma: no cover
        return self.subscribe(Breakpoint(**kwargs))

    def breakword(self, **kwargs):  # pragma: no cover
        return self.subscribe(Breakpoint(use_breakword=True, **kwargs))

    def display(self, *, breakword=False, word=None, **kwargs):
        sub = self.subscribe(Displayer(**kwargs))
        if breakword:  # pragma: no cover
            self.breakword(word=word)
        return sub

    def give(self, *keys, **extra):
        from .core import Giver

        giver = Giver(keys=keys, extra=extra)

        if len(keys) == 0:
            gv = giver.produce
        elif len(keys) == 1:
            gv = giver
        else:

            def gv(x):
                return giver(*x)

        return self.subscribe(gv)

    def ksubscribe(self, fn):
        fn = lax_function(fn)
        self.subscribe(lambda data: fn(**data))

    def kwrap(self, fn, begin="$begin", end="$end"):
        return self.wrap(fn, begin=begin, end=end, pass_keys=True)

    def print(self, format=None):
        obs = self
        if format is not None:
            obs = self.format(format)
        return obs.subscribe(print)

    def wrap(self, fn, begin="$begin", end="$end", pass_keys=False):
        if pass_keys:
            fn = lax_function(fn)

        if inspect.isgeneratorfunction(fn):
            fn = contextmanager(fn)

        managers = {}

        def watch(data):
            if begin in data:
                key = data[begin]
                assert key not in managers
                if hasattr(fn, "__enter__"):
                    manager = fn
                elif pass_keys:
                    manager = fn(**data)
                else:
                    manager = fn()

                managers[key] = manager
                manager.__enter__()

            if end in data:
                key = data[end]
                managers[key].__exit__(None, None, None)
                del managers[key]

        return self.subscribe(watch)
