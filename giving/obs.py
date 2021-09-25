import inspect
from contextlib import contextmanager
from functools import partial, wraps

from . import operators as op
from .executors import Breakpoint, Displayer
from .utils import lax_function


def _opmethod(operator):
    @wraps(operator)
    def f(self, *args, **kwargs):
        return self.pipe(operator(*args, **kwargs))

    return f


class Failure(Exception):
    pass


class ObservableProxy:
    """Wraps an Observable_ to provide a richer interface.

    .. _Observable: https://rxpy.readthedocs.io/en/latest/reference_observable.html

    For convenience, all operators in :mod:`giving.operators` are
    provided as methods on these objects.
    """

    def __init__(self, obs):
        assert not isinstance(obs, ObservableProxy)
        self.obs = obs

    def copy(self, new_obs):
        """Copy this observable with a new underlying observable."""
        return type(self)(new_obs)

    ###################
    # Wrapped methods #
    ###################

    def pipe(self, *args, **kwargs):
        """Pipe one or more operators.

        Returns: An ObservableProxy.
        """
        return self.copy(self.obs.pipe(*args, **kwargs))

    def subscribe(self, *args, **kwargs):
        """Subscribe a function to this Observable stream.

        .. code-block:: python

            with given() as gv:
                gv.subscribe(print)

                results = []
                gv["x"].subscribe(results.append)

                give(x=1)  # prints {"x": 1}
                give(x=2)  # prints {"x": 2}

                assert results == [1, 2]

        Arguments:
            observer:
                The object that is to receive notifications.
            on_error:
                Action to invoke upon exceptional termination of the
                observable sequence.
            on_completed:
                 Action to invoke upon graceful termination of the
                 observable sequence.
            on_next:
                Action to invoke for each element in the observable
                sequence.

        Returns:
            An object representing the subscription with a ``dispose()``
            method to remove it.
        """
        return self.obs.subscribe(*args, **kwargs)

    def subscribe_(self, *args, **kwargs):
        return self.obs.subscribe_(*args, **kwargs)

    ###################
    # Extra utilities #
    ###################

    def accum(self, obj=None):
        """Accumulate into a list or set.

        Arguments:
            obj: The object in which to accumulate, either a list or
                a set. If not provided, a new list is created.

        Returns:
            The object in which the values will be accumulated.
        """
        if obj is None:
            obj = []
            method = obj.append
        elif isinstance(obj, list):
            method = obj.append
        elif isinstance(obj, set):
            method = obj.add
        else:
            raise TypeError("Argument to accum must be a list or set.")

        self.subscribe(method)
        return obj

    def breakpoint(self, **kwargs):  # pragma: no cover
        """Trigger a breakpoint on every entry.

        Arguments:
            skip:
                A list of globs corresponding to modules to skip during debugging,
                for example ``skip=["giving.*"]`` would skip all frames that are in
                the ``giving`` module.
        """
        return self.subscribe(Breakpoint(**kwargs))

    def breakword(self, **kwargs):  # pragma: no cover
        """Trigger a breakpoint using ``breakword``.

        This feature requires the ``breakword`` package to be installed, and the
        :func:`~giving.operators.tag` operator to be applied.

        .. code-block:: python

            gvt = gv.tag()
            gvt.display()
            gvt.breakword()

        The above will display words next to each entry. Set the BREAKWORD environment
        to one of these words to set a breakpoint when it is printed.

        Arguments:
            skip:
                A list of globs corresponding to modules to skip during debugging,
                for example ``skip=["giving.*"]`` would skip all frames that are in
                the ``giving`` module.
            word:
                Only trigger the breakpoint on the given word.
        """
        return self.subscribe(Breakpoint(use_breakword=True, **kwargs))

    def display(self, *, breakword=None, skip=[], **kwargs):
        """Pretty-print each element.

        Arguments:
            colors:
                Whether to colorize the output or not.
            time_format:
                How to format the time (if present), e.g. ``"%Y-%m-%d %H:%M:%S"``
            breakword:
                If not None, run ``self.breakword(word=breakword)``.
            skip:
                If ``breakword is not None``, pass ``skip`` to the debugger.
        """
        sub = self.subscribe(Displayer(**kwargs))
        if breakword:  # pragma: no cover
            self.breakword(word=breakword, skip=skip)
        return sub

    def fail(self, exc_type=Failure):
        """Raise an exception if the stream produces anything.

        Arguments:
            exc_type:
                The exception type to raise. Will be passed the next data
                element, and the result is raised. Defaults to
                :class:`~giving.obs.Failure`.
        """

        def _fail(data):
            raise exc_type(data)

        return self.subscribe(_fail)

    def fail_if_empty(self, exc_type=Failure):
        """Raise an exception if the stream is empty.

        Arguments:
            exc_type:
                The exception type to raise. Defaults to :class:`~giving.obs.Failure`.
        """

        def _fail(is_empty):
            if is_empty:
                raise exc_type(is_empty)

        return self.is_empty().subscribe(_fail)

    def give(self, *keys, **extra):
        """``give`` each element.

        Be careful using this method because it could easily lead to an infinite loop.

        Arguments:
            keys: Key(s) under which to give the elements.
            extra: Extra key/value pairs to give along with the rest.
        """
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
        """Subscribe a function called with keyword arguments.

        .. note::
            The function passed to ``ksubscribe`` is wrapped with
            :func:`~giving.utils.lax_function`, so it is not necessary
            to add a ``**kwargs`` argument for keys that you do not need.

        .. code-block:: python

            gv.ksubscribe(lambda x, y=None, z=None: print(x, y, z))
            give(x=1, z=2, abc=3)  # Prints 1, None, 2

        Arguments:
            fn: The function to call.
        """
        fn = lax_function(fn)
        self.subscribe(lambda data: fn(**data))

    def kwrap(self, name, fn=None, return_function=False):
        """Subscribe a context manager, corresponding to :meth:`~giving.core.Giver.wrap`.

        ``obs.kwrap(fn)`` is shorthand for ``obs.wrap(fn, pass_keys=True)``.

        .. note::
            The function passed to ``ksubscribe`` is wrapped with
            :func:`~giving.utils.lax_function`, so it is not necessary
            to add a ``**kwargs`` argument for keys that you do not need.

        .. code-block:: python

            @gv.kwrap
            def _(x):
                print(">", x)
                yield
                print("<", x)

            with give.wrap(x=1):      # prints >1
                ...
                with give.wrap(x=2):  # prints >2
                    ...
                ...                   # prints <2
            ...                       # prints <1

        Arguments:
            name: The name of the wrap block to subscribe to.
            fn: The wrapper function. The arguments to ``give.wrap`` are transferred to
                this function as keyword arguments.
        """
        return self.wrap(name, fn, pass_keys=True, return_function=return_function)

    def print(self, format=None, skip_missing=False):
        """Print each element of the stream.

        Arguments:
            format: A format string as would be used with ``str.format``.
            skip_missing: Whether to ignore KeyErrors due to missing entries in the format.
        """
        obs = self
        if format is not None:
            obs = self.format(format, skip_missing=skip_missing)
        return obs.subscribe(print)

    def wrap(self, name, fn=None, pass_keys=False, return_function=False):
        """Subscribe a context manager, corresponding to :meth:`~giving.core.Giver.wrap`.

        .. code-block:: python

            @gv.wrap("main")
            def _():
                print("<")
                yield
                print(">")

            with give.wrap("main"):    # prints <
                ...
                with give.wrap("sub"):
                    ...
                ...
            ...                        # prints >

        Arguments:
            name: The name of the wrap block to subscribe to.
            fn: The wrapper function OR an object with an ``__enter__`` method. If the wrapper is
                a generator, it will be wrapped with ``contextmanager(fn)``.
                If a function, it will be called with no arguments, or with the arguments given to
                ``give.wrap`` if ``pass_keys=True``.
            pass_keys: Whether to pass the arguments to ``give.wrap`` to this function as keyword
                arguments. You may use :meth:`kwrap` as a shortcut to ``pass_keys=True``.
        """
        if fn is None:
            return partial(self.wrap, name, pass_keys=pass_keys, return_function=True)

        if pass_keys:
            fn = lax_function(fn)

        if inspect.isgeneratorfunction(fn):
            fn = contextmanager(fn)

        managers = {}

        def watch(data):
            wr = data.get("$wrap", None)
            if wr is None or wr["name"] is not name:
                return

            if wr["step"] == "begin":
                key = wr["id"]
                assert key not in managers
                if hasattr(fn, "__enter__"):
                    manager = fn
                elif pass_keys:
                    manager = fn(**data)
                else:
                    manager = fn()

                managers[key] = manager
                manager.__enter__()

            if wr["step"] == "end":
                key = wr["id"]
                managers[key].__exit__(None, None, None)
                del managers[key]

        disposable = self.subscribe(watch)
        if return_function:
            return fn
        else:
            return disposable

    def __or__(self, other):
        """Alias for :func:`~giving.operators.merge`.

        Merge this ObservableProxy with another.
        """
        return self.merge(other)

    __ror__ = __or__

    def __rshift__(self, subscription):
        """Alias for :meth:`subscribe`.

        If ``subscription`` is a list or a set, accumulate into it.
        """
        if isinstance(subscription, list):
            subscription = subscription.append
        elif isinstance(subscription, set):
            subscription = subscription.add
        return self.subscribe(subscription)

    def __getitem__(self, item):
        """Mostly an alias for :meth:`~giving.operators.getitem`.

        Extra feature: if the item starts with ``"?"``, ``getitem`` is called with
        ``strict=False``.
        """
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
    catch = _opmethod(op.catch)
    combine_latest = _opmethod(op.combine_latest)
    concat = _opmethod(op.concat)
    contains = _opmethod(op.contains)
    count = _opmethod(op.count)
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
    # max_by = _opmethod(op.max_by)
    merge = _opmethod(op.merge)
    merge_all = _opmethod(op.merge_all)
    min = _opmethod(op.min)
    # min_by = _opmethod(op.min_by)
    multicast = _opmethod(op.multicast)
    observe_on = _opmethod(op.observe_on)
    on_error_resume_next = _opmethod(op.on_error_resume_next)
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
    timeout = _opmethod(op.timeout)
    timeout_with_mapper = _opmethod(op.timeout_with_mapper)
    timestamp = _opmethod(op.timestamp)
    to_dict = _opmethod(op.to_dict)
    to_future = _opmethod(op.to_future)
    to_iterable = _opmethod(op.to_iterable)
    to_list = _opmethod(op.to_list)
    to_marbles = _opmethod(op.to_marbles)
    to_set = _opmethod(op.to_set)
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
    augment = _opmethod(op.augment)
    as_ = _opmethod(op.as_)
    average_and_variance = _opmethod(op.average_and_variance)
    collect_between = _opmethod(op.collect_between)
    format = _opmethod(op.format)
    getitem = _opmethod(op.getitem)
    group_wrap = _opmethod(op.group_wrap)
    keep = _opmethod(op.keep)
    kfilter = _opmethod(op.kfilter)
    kmap = _opmethod(op.kmap)
    kmerge = _opmethod(op.kmerge)
    kscan = _opmethod(op.kscan)
    norepeat = _opmethod(op.norepeat)
    roll = _opmethod(op.roll)
    sole = _opmethod(op.sole)
    tag = _opmethod(op.tag)
    throttle = _opmethod(op.throttle)
    variance = _opmethod(op.variance)
    where = _opmethod(op.where)
    where_any = _opmethod(op.where_any)
