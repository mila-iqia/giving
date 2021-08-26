from itertools import count

from . import operators as op
from .executors import Breakpoint, Displayer


def _opmethod(name, operator):
    def f(self, *args, **kwargs):
        return self.pipe(operator(*args, **kwargs))

    f.__name__ = name
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
        strict = not any("?" in x for x in item)
        item = [x.lstrip("?") for x in item]
        return self.getitem(*item, strict=strict)

    all = _opmethod("all", op.all)
    amb = _opmethod("amb", op.amb)
    as_observable = _opmethod("as_observable", op.as_observable)
    average = _opmethod("average", op.average)
    buffer = _opmethod("buffer", op.buffer)
    buffer_toggle = _opmethod("buffer_toggle", op.buffer_toggle)
    buffer_when = _opmethod("buffer_when", op.buffer_when)
    buffer_with_count = _opmethod("buffer_with_count", op.buffer_with_count)
    buffer_with_time = _opmethod("buffer_with_time", op.buffer_with_time)
    buffer_with_time_or_count = _opmethod(
        "buffer_with_time_or_count", op.buffer_with_time_or_count
    )
    cast = _opmethod("cast", op.cast)
    catch = _opmethod("catch", op.catch)
    combine_latest = _opmethod("combine_latest", op.combine_latest)
    concat = _opmethod("concat", op.concat)
    contains = _opmethod("contains", op.contains)
    count = _opmethod("count", op.count)
    datetime = _opmethod("datetime", op.datetime)
    debounce = _opmethod("debounce", op.debounce)
    default_if_empty = _opmethod("default_if_empty", op.default_if_empty)
    delay = _opmethod("delay", op.delay)
    delay_subscription = _opmethod("delay_subscription", op.delay_subscription)
    delay_with_mapper = _opmethod("delay_with_mapper", op.delay_with_mapper)
    dematerialize = _opmethod("dematerialize", op.dematerialize)
    distinct = _opmethod("distinct", op.distinct)
    distinct_until_changed = _opmethod(
        "distinct_until_changed", op.distinct_until_changed
    )
    do = _opmethod("do", op.do)
    do_action = _opmethod("do_action", op.do_action)
    do_while = _opmethod("do_while", op.do_while)
    element_at = _opmethod("element_at", op.element_at)
    element_at_or_default = _opmethod("element_at_or_default", op.element_at_or_default)
    exclusive = _opmethod("exclusive", op.exclusive)
    expand = _opmethod("expand", op.expand)
    filter = _opmethod("filter", op.filter)
    filter_indexed = _opmethod("filter_indexed", op.filter_indexed)
    finally_action = _opmethod("finally_action", op.finally_action)
    find = _opmethod("find", op.find)
    find_index = _opmethod("find_index", op.find_index)
    first = _opmethod("first", op.first)
    first_or_default = _opmethod("first_or_default", op.first_or_default)
    flat_map = _opmethod("flat_map", op.flat_map)
    flat_map_indexed = _opmethod("flat_map_indexed", op.flat_map_indexed)
    flat_map_latest = _opmethod("flat_map_latest", op.flat_map_latest)
    fork_join = _opmethod("fork_join", op.fork_join)
    group_by = _opmethod("group_by", op.group_by)
    group_by_until = _opmethod("group_by_until", op.group_by_until)
    group_join = _opmethod("group_join", op.group_join)
    ignore_elements = _opmethod("ignore_elements", op.ignore_elements)
    is_empty = _opmethod("is_empty", op.is_empty)
    join = _opmethod("join", op.join)
    last = _opmethod("last", op.last)
    last_or_default = _opmethod("last_or_default", op.last_or_default)
    map = _opmethod("map", op.map)
    map_indexed = _opmethod("map_indexed", op.map_indexed)
    materialize = _opmethod("materialize", op.materialize)
    max = _opmethod("max", op.max)
    max_by = _opmethod("max_by", op.max_by)
    merge = _opmethod("merge", op.merge)
    merge_all = _opmethod("merge_all", op.merge_all)
    min = _opmethod("min", op.min)
    min_by = _opmethod("min_by", op.min_by)
    multicast = _opmethod("multicast", op.multicast)
    observe_on = _opmethod("observe_on", op.observe_on)
    on_error_resume_next = _opmethod("on_error_resume_next", op.on_error_resume_next)
    overload = _opmethod("overload", op.overload)
    pairwise = _opmethod("pairwise", op.pairwise)
    partition = _opmethod("partition", op.partition)
    partition_indexed = _opmethod("partition_indexed", op.partition_indexed)
    pluck = _opmethod("pluck", op.pluck)
    pluck_attr = _opmethod("pluck_attr", op.pluck_attr)
    publish = _opmethod("publish", op.publish)
    publish_value = _opmethod("publish_value", op.publish_value)
    reduce = _opmethod("reduce", op.reduce)
    ref_count = _opmethod("ref_count", op.ref_count)
    repeat = _opmethod("repeat", op.repeat)
    replay = _opmethod("replay", op.replay)
    retry = _opmethod("retry", op.retry)
    sample = _opmethod("sample", op.sample)
    scan = _opmethod("scan", op.scan)
    sequence_equal = _opmethod("sequence_equal", op.sequence_equal)
    share = _opmethod("share", op.share)
    single = _opmethod("single", op.single)
    single_or_default = _opmethod("single_or_default", op.single_or_default)
    single_or_default_async = _opmethod(
        "single_or_default_async", op.single_or_default_async
    )
    skip = _opmethod("skip", op.skip)
    skip_last = _opmethod("skip_last", op.skip_last)
    skip_last_with_time = _opmethod("skip_last_with_time", op.skip_last_with_time)
    skip_until = _opmethod("skip_until", op.skip_until)
    skip_until_with_time = _opmethod("skip_until_with_time", op.skip_until_with_time)
    skip_while = _opmethod("skip_while", op.skip_while)
    skip_while_indexed = _opmethod("skip_while_indexed", op.skip_while_indexed)
    skip_with_time = _opmethod("skip_with_time", op.skip_with_time)
    slice = _opmethod("slice", op.slice)
    some = _opmethod("some", op.some)
    starmap = _opmethod("starmap", op.starmap)
    starmap_indexed = _opmethod("starmap_indexed", op.starmap_indexed)
    start_with = _opmethod("start_with", op.start_with)
    subscribe_on = _opmethod("subscribe_on", op.subscribe_on)
    sum = _opmethod("sum", op.sum)
    switch_latest = _opmethod("switch_latest", op.switch_latest)
    take = _opmethod("take", op.take)
    take_last = _opmethod("take_last", op.take_last)
    take_last_buffer = _opmethod("take_last_buffer", op.take_last_buffer)
    take_last_with_time = _opmethod("take_last_with_time", op.take_last_with_time)
    take_until = _opmethod("take_until", op.take_until)
    take_until_with_time = _opmethod("take_until_with_time", op.take_until_with_time)
    take_while = _opmethod("take_while", op.take_while)
    take_while_indexed = _opmethod("take_while_indexed", op.take_while_indexed)
    take_with_time = _opmethod("take_with_time", op.take_with_time)
    throttle_first = _opmethod("throttle_first", op.throttle_first)
    throttle_with_mapper = _opmethod("throttle_with_mapper", op.throttle_with_mapper)
    throttle_with_timeout = _opmethod("throttle_with_timeout", op.throttle_with_timeout)
    time_interval = _opmethod("time_interval", op.time_interval)
    timedelta = _opmethod("timedelta", op.timedelta)
    timeout = _opmethod("timeout", op.timeout)
    timeout_with_mapper = _opmethod("timeout_with_mapper", op.timeout_with_mapper)
    timestamp = _opmethod("timestamp", op.timestamp)
    to_dict = _opmethod("to_dict", op.to_dict)
    to_future = _opmethod("to_future", op.to_future)
    to_iterable = _opmethod("to_iterable", op.to_iterable)
    to_list = _opmethod("to_list", op.to_list)
    to_marbles = _opmethod("to_marbles", op.to_marbles)
    to_set = _opmethod("to_set", op.to_set)
    typing = _opmethod("typing", op.typing)
    while_do = _opmethod("while_do", op.while_do)
    window = _opmethod("window", op.window)
    window_toggle = _opmethod("window_toggle", op.window_toggle)
    window_when = _opmethod("window_when", op.window_when)
    window_with_count = _opmethod("window_with_count", op.window_with_count)
    window_with_time = _opmethod("window_with_time", op.window_with_time)
    window_with_time_or_count = _opmethod(
        "window_with_time_or_count", op.window_with_time_or_count
    )
    with_latest_from = _opmethod("with_latest_from", op.with_latest_from)
    zip = _opmethod("zip", op.zip)
    zip_with_iterable = _opmethod("zip_with_iterable", op.zip_with_iterable)
    zip_with_list = _opmethod("zip_with_list", op.zip_with_list)

    affix = _opmethod("affix", op.affix)
    as_ = _opmethod("as_", op.as_)
    collect_between = _opmethod("collect_between", op.collect_between)
    getitem = _opmethod("getitem", op.getitem)
    stream_once = _opmethod("stream_once", op.stream_once)
    tag = _opmethod("tag", op.tag)
    unique = _opmethod("unique", op.unique)
    where = _opmethod("where", op.where)

    def display(self, *, breakword=False, word=None, **kwargs):
        sub = self.subscribe(Displayer(**kwargs))
        if breakword:
            self.breakword(word=word)
        return sub

    def breakpoint(self):
        return self.subscribe(Breakpoint())

    def breakword(self, **kwargs):
        return self.subscribe(Breakpoint(use_breakword=True, **kwargs))

    def give(self, *keys, **extra):
        from .core import giver

        return self.subscribe(giver(*keys, **extra))
