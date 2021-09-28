import inspect
from contextlib import contextmanager
from functools import partial, wraps

import rx

from . import operators
from .executors import Breakpoint, Displayer
from .gvr import Giver, global_context
from .utils import lax_function


class Failure(Exception):
    """Error type raised by fail() by default."""


class ExtendedInterface:
    """Defines a rich interface for an Observable."""

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
                :class:`~giving.gvn.Failure`.
        """

        def _fail(data):
            raise exc_type(data)

        return self.subscribe(_fail)

    def fail_if_empty(self, exc_type=Failure):
        """Raise an exception if the stream is empty.

        Arguments:
            exc_type:
                The exception type to raise. Defaults to :class:`~giving.gvn.Failure`.
        """

        def _fail(is_empty):
            if is_empty:
                raise exc_type(is_empty)

        return self.is_empty().subscribe(_fail)

    def give(self, *keys, **extra):
        """Give each element.

        This calls :func:`~giving.core.give` for each value in the stream.

        Be careful using this method because it could easily lead to an infinite loop.

        Arguments:
            keys: Key(s) under which to give the elements.
            extra: Extra key/value pairs to give along with the rest.
        """
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

    #################################################
    # Methods for each operator in giving.operators #
    #################################################

    ...  # The methods are set with _put_opmethods below


def _opmethod(operator):
    @wraps(operator)
    def f(self, *args, **kwargs):
        return self.pipe(operator(*args, **kwargs))

    return f


def _put_opmethods(cls, names):
    for name in names:
        operator = getattr(operators, name)
        setattr(cls, name, _opmethod(operator))


_put_opmethods(ExtendedInterface, operators.__all__)


class ObservableProxy(ExtendedInterface):
    """Wraps an Observable_ to provide a richer interface.

    .. _Observable: https://rxpy.readthedocs.io/en/latest/reference_observable.html

    For convenience, all operators in :mod:`giving.operators` are
    provided as methods on these objects.
    """

    def __init__(self, obs):
        assert not isinstance(obs, ObservableProxy)
        self.obs = obs

    def _copy(self, new_obs):
        """Copy this observable with a new underlying observable."""
        return type(self)(new_obs)

    ###################
    # Wrapped methods #
    ###################

    def pipe(self, *args, **kwargs):
        """Pipe one or more operators.

        Returns: An ObservableProxy.
        """
        return self._copy(self.obs.pipe(*args, **kwargs))

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


class Stream:
    def __init__(self):
        self.observers = []

        def make(observer, scheduler):
            self.observers.append(observer)

        src = rx.create(make)
        self.source = ObservableProxy(src)

    def push(self, data):
        for obs in self.observers:
            obs.on_next(data)

    def complete(self):
        for obs in self.observers:
            obs.on_completed()


class Given:
    """Context manager that yields an ObservableProxy for a block.

    Upon entering, an :class:`~giving.gvn.ObservableProxy` is yielded,
    and calls to ``give`` will trigger that Observable. Upon exiting,
    the Observable is marked as completed, triggering reductions such
    as ``min`` or ``sum``.

    Arguments:
        key: The key to extract, or None.
        context:
            The ContextVar to use to sync with ``give``.
    """

    def __init__(self, context=global_context):
        self._token = self._stream = None
        self._context = context

    def __enter__(self):
        assert self._token is None
        self._stream = strm = Stream()
        h = self._context.get()
        self._token = self._context.set((*h, strm.push))
        return strm.source

    def __exit__(self, exc_type=None, exc=None, tb=None):
        self._stream.complete()
        self._context.reset(self._token)
        self._token = self._stream = None
