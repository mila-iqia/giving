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
            if hasattr(co, "replace"):
                newco = co.replace(
                    co_flags=co.co_flags | KWVAR_FLAG,
                    # Add a dummy keyword argument with an illegal name
                    co_varnames=(*co.co_varnames, "#"),
                )
            else:  # pragma: no cover
                newco = types.CodeType(
                    co.co_argcount,
                    co.co_kwonlyargcount,
                    co.co_nlocals,
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
