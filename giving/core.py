import ast
import sys
import time
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from types import SimpleNamespace

import rx
from varname import ImproperUseError, VarnameRetrievingError, argname, varname
from varname.utils import get_node

from . import operators as op
from .obs import ObservableProxy

ABSENT = object()

global_context = ContextVar("global_context", default=())
global_inherited = ContextVar("global_inherited", default={})

_block_classes = {
    ast.If: ("body", "orelse"),
    ast.For: ("body", "orelse"),
    ast.While: ("body", "orelse"),
    ast.FunctionDef: ("body",),
    ast.AsyncFunctionDef: ("body",),
    ast.With: ("body",),
    ast.AsyncWith: ("body",),
    ast.AsyncFor: ("body", "orelse"),
}


_improper_nullary_give_error = (
    "give() with no arguments must immediately follow an assignment"
)


special_keys = {}


global_count = count(0)


def register_special(key):
    def deco(func):
        special_keys[key] = func
        return func

    return deco


@register_special("$time")
def _special_time():
    return time.time()


@register_special("$frame")
def _special_frame():
    return sys._getframe(3)


LinePosition = namedtuple("LinePosition", ["name", "filename", "lineno"])


@register_special("$line")
def _special_line():
    fr = sys._getframe(3)
    co = fr.f_code
    return LinePosition(co.co_name, co.co_filename, fr.f_lineno)


def _find_above(frame):
    node = get_node(frame + 1)
    if node is None:
        raise VarnameRetrievingError(
            "Cannot retrieve the node where the function is called"
        )

    while node.parent is not None:
        parent = node.parent
        fields = _block_classes.get(type(parent), None)
        if fields is None:
            node = parent
            continue
        else:
            for field in fields:
                f = getattr(parent, field)
                if node in f:
                    idx = f.index(node)
                    if idx == 0:
                        raise ImproperUseError(_improper_nullary_give_error)

                    assignment = f[idx - 1]

                    if isinstance(assignment, ast.Assign):
                        target = assignment.targets[-1]
                        names = [target.id]
                    elif isinstance(assignment, (ast.AugAssign, ast.AnnAssign)):
                        names = [assignment.target.id]
                    else:
                        raise ImproperUseError(_improper_nullary_give_error)

                    fr = sys._getframe(frame)
                    rval = {}

                    for name in names:
                        if name in fr.f_locals:
                            rval[name] = fr.f_locals[name]
                        elif name in fr.f_globals:
                            rval[name] = fr.f_globals[name]
                        else:  # pragma: no cover
                            # I am not sure how to trigger this
                            raise Exception("Could not resolve value")
                    return rval

            else:  # pragma: no cover
                # I am not sure how to trigger this
                raise Exception("Could not find node position")

    # I am not sure how to trigger this
    raise Exception("Could not find node")  # pragma: no cover


def resolve(frame, func, args):
    """Return a {variable_name: value} dictionary depending on usage.

    * (len(args) == 0) => Use the variable assigned in the line before the call.
    * (len(args) == 1) => Use the variable the call is assigned to.
    * (len(args) >= 1) => Use the variables passed as arguments to the call.
    """
    nargs = len(args)

    if nargs == 0:
        return _find_above(frame=frame + 2)

    if nargs == 1:
        try:
            assigned_to = varname(frame=frame + 1, strict=True, raise_exc=False)
        except ImproperUseError:
            assigned_to = None
        if assigned_to is not None:
            return {assigned_to: args[0]}

    argnames = argname("args", func=func, frame=frame + 1, vars_only=False)
    if argnames is None:  # pragma: no cover
        # I am not sure how to trigger this
        raise Exception("Could not resolve arg names")

    return {name: value for name, value in zip(argnames, args)}


class Giver:
    def __init__(
        self,
        *,
        keys=None,
        special=[],
        extra={},
        context=global_context,
        inherited=global_inherited,
    ):
        self.keys = keys
        self.special = special
        self.extra = extra
        self.context = context
        self.inherited = inherited

    @property
    def line(self):
        return Giver(
            keys=self.keys,
            special=(*self.special, "$line"),
            extra=self.extra,
            context=self.context,
            inherited=self.inherited,
        )

    @property
    def time(self):
        return Giver(
            keys=self.keys,
            special=(*self.special, "$time"),
            extra=self.extra,
            context=self.context,
            inherited=self.inherited,
        )

    @contextmanager
    def inherit(self, **keys):
        inh = self.inherited.get()
        token = self.inherited.set({**inh, **keys})
        try:
            yield
        finally:
            self.inherited.reset(token)

    @contextmanager
    def wrap(self, **keys):
        num = next(global_count)
        self.produce({"$begin": num, **keys})
        try:
            yield
        finally:
            self.produce({"$end": num, **keys})

    @contextmanager
    def wrap_inherit(self, **keys):
        with self.wrap(**keys):
            with self.inherit(**keys):
                yield

    def produce(self, values):
        for special in self.special:
            values[special] = special_keys[special]()

        if self.extra:
            values = {**self.extra, **values}

        inh = self.inherited.get()
        if inh is not None:
            values = {**inh, **values}

        for handler in self.context.get():
            handler(values)

    def __call__(self, *args, **values):
        h = self.context.get()
        if h:
            if self.keys:
                if len(args) != len(self.keys):
                    raise ImproperUseError(
                        f"Giver for {self.keys} must have {len(self.keys)} positional argument(s)."
                    )
                keyed = dict(zip(self.keys, args))
                values = {**keyed, **values}
            elif args:
                values = {**resolve(1, self, args), **values}
            elif not values:
                values = resolve(1, self, ())

            self.produce(values)

        if len(args) == 1:
            return args[0]
        else:
            return None


def giver(*keys, **extra):
    normal = [k for k in keys if not k.startswith("$")]
    special = [k for k in keys if k.startswith("$")]
    return Giver(keys=normal, special=special, extra=extra)


class Given:
    def __init__(self, key=None, context=global_context):
        self.key = key
        self.token = self.observers = None
        self.context = context

    def __enter__(self):
        self.observers = []

        def make(observer, scheduler):
            self.observers.append(observer)

        src = rx.create(make)

        def handler(values):
            for obs in self.observers:
                obs.on_next(values)

        h = self.context.get()
        self.token = self.context.set((*h, handler))

        if isinstance(self.key, str):
            src = src.pipe(op.getitem(self.key))

        return ObservableProxy(src)

    def __exit__(self, exc_type=None, exc=None, tb=None):
        for obs in self.observers:
            obs.on_completed()
        self.context.reset(self.token)
        self.token = self.observers = None


def make_give(context=None):
    context = context or ContextVar("context", default=())
    give = Giver(context=context)

    def given(*args, **kwargs):
        return Given(*args, **kwargs, context=context)

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
