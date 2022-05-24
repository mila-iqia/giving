import fnmatch
import functools
import sys
import types

from reactivex import operators as rxop
from reactivex.operators import NotSet


def keyword_decorator(deco):
    """Wrap a decorator to optionally takes keyword arguments."""

    @functools.wraps(deco)
    def new_deco(fn=None, **kwargs):
        if fn is None:

            @functools.wraps(deco)
            def newer_deco(fn):
                return deco(fn, **kwargs)

            return newer_deco
        else:
            return deco(fn, **kwargs)

    return new_deco


def lax_function(fn):
    """Add a ``**kwargs`` argument to fn if it does not have one.

    ``fn`` is not modified. If it has a varkw argument, ``fn`` is returned directly,
    otherwise a new function is made that takes varkw and does nothing with them (it
    just drops them)

    Example:

        .. code-block:: python

            @lax_function
            def f(x):
                return x * x

            f(x=4, y=123, z="blah")  # works, returns 16. y and z are ignored.
    """
    # NOTE: The purpose of ``lax_function`` is to simplify ``kmap``, ``kfilter``, ``ksubscribe``
    # and other mappers that can be partially applied to the element dictionaries in an
    # observable stream.

    if isinstance(fn, types.FunctionType):
        KWVAR_FLAG = 8
        co = fn.__code__
        if not co.co_flags & KWVAR_FLAG:
            if hasattr(co, "replace"):
                newco = co.replace(
                    # Let's lie and pretend it has **kwargs
                    co_flags=co.co_flags | KWVAR_FLAG,
                    # Add a dummy keyword argument with an illegal name
                    co_varnames=(*co.co_varnames, "#"),
                    # That dummy argument is a new local
                    co_nlocals=co.co_nlocals + 1,
                )
            else:  # pragma: no cover
                newco = types.CodeType(
                    co.co_argcount,
                    co.co_kwonlyargcount,
                    co.co_nlocals + 1,
                    co.co_stacksize,
                    co.co_flags | KWVAR_FLAG,
                    co.co_code,
                    co.co_consts,
                    co.co_names,
                    (*co.co_varnames, "#"),
                    co.co_filename,
                    co.co_name,
                    co.co_firstlineno,
                    co.co_lnotab,
                    co.co_freevars,
                    co.co_cellvars,
                )

            newfn = types.FunctionType(
                name=fn.__name__,
                code=newco,
                globals=fn.__globals__,
                closure=fn.__closure__,
            )
            newfn.__defaults__ = fn.__defaults__
            newfn.__kwdefaults__ = fn.__kwdefaults__
            return newfn

    return fn


class _Reducer:
    def __init__(self, reduce, roll):
        self.reduce = reduce
        self.roll = roll


@keyword_decorator
def reducer(func, default_seed=NotSet, postprocess=NotSet):
    """Create a reduction operator.

    .. note::
        If ``func`` is not given, this returns a decorator.

    * If func is a function, it should have the signature ``func(accum, x)``
      where ``accum`` is the last result and ``x`` is the new data.
    * If func is a class, it should have the following methods:

      * ``reduce(accum, x)``, used when ``scan`` is boolean (``scan=False`` or ``scan=True``)
      * ``roll(accum, x, drop, last_size, current_size)`` (see :func:`giving.operators.roll`),
        used when ``scan`` is an integer (``scan=n``). It should implement an optimized way to
        reduce over the last n elements of a sequence. Note that ``last_size`` and ``current_size``
        cannot exceed ``n``.
      * Any arguments to ``__init__`` will be passed over when the operator is called.

    Example:

    .. code-block:: python

        @reducer
        def sum(last, new):
            return last + new

    .. code-block:: python

        @reducer
        class sum:
            def reduce(self, last, new):
                return last + new

            def roll(self, last, new, drop, last_size, current_size):
                result = last + new
                if last_size == current_size:
                    result -= drop
                return result

    Arguments:
        func: A function or class that defines the reduction.
        default_seed: The default seed to start the reduction.
        postprocess: A postprocessing step to apply to the result of
            the operator.

    Returns:
        A reduction operator that takes a ``scan`` argument to perform
        a scan or roll operation instead of reduce.
    """

    from .extraops import roll

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
            oper = rxop.compose(oper, postprocess)

        return oper

    _create.__name__ = name
    _create.__doc__ = func.__doc__
    return _create


def reduced_traceback(skip=["giving.*", "rx.*"], depth=2):
    """Create a traceback from the current frame, minus skipped modules.

    Arguments:
        skip: List of modules to skip.
        depth: Depth at which to start.
    """
    # Adapted from one of the answers on here (plus skipping code):
    # https://stackoverflow.com/questions/27138440/how-to-create-a-traceback-object

    tb = None
    while True:
        try:
            frame = sys._getframe(depth)
            depth += 1
        except ValueError:
            break

        name = frame.f_globals.get("__name__", "")
        if any(fnmatch.fnmatch(name, sk) for sk in skip):
            continue
        tb = types.TracebackType(tb, frame, frame.f_lasti, frame.f_lineno)

    return tb
