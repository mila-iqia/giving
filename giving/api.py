from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from types import SimpleNamespace

from .gvn import Given
from .gvr import Giver, global_context


def make_give(context=None):
    """Create independent give/given/accumulate.

    The resulting functions share their own ``ContextVar``, which
    makes them independent from the main instances of ``give`` and
    ``given``.

    Arguments:
        context: The ``ContextVar`` set by ``given`` and used by ``give``,
            or ``None`` if a new ``ContextVar`` is to be created.

    Returns:
        A SimpleNamespace with attributes ``context``, ``give``,
        ``given`` and ``accumulate``.
    """

    context = context or ContextVar("context", default=())
    give = Giver(context=context)
    given = partial(Given, context=context)

    @contextmanager
    def accumulate(key=None):
        results = []
        with given(key) as gv:
            gv.subscribe(results.append)
            yield results

    return SimpleNamespace(
        context=context,
        give=give,
        given=given,
        accumulate=accumulate,
    )


_global_given = make_give(context=global_context)

give = _global_given.give
given = _global_given.given
accumulate = _global_given.accumulate
