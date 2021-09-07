import functools
import types


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
    if isinstance(fn, types.FunctionType):
        KWVAR_FLAG = 8
        co = fn.__code__
        if not co.co_flags & KWVAR_FLAG:
            newfn = types.FunctionType(
                name=fn.__name__,
                code=co.replace(
                    co_flags=co.co_flags | KWVAR_FLAG,
                    # Add a dummy keyword argument with an illegal name
                    co_varnames=(*co.co_varnames, "#"),
                ),
                globals=fn.__globals__,
                closure=fn.__closure__,
            )
            newfn.__defaults__ = fn.__defaults__
            newfn.__kwdefaults__ = fn.__kwdefaults__
            return newfn

    return fn
