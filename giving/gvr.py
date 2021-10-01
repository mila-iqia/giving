import ast
import sys
import time
from collections import namedtuple
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from varname import ImproperUseError, VarnameRetrievingError, argname, varname
from varname.utils import get_node

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
    """Return a decorator to register a function for a special key.

    The function is called with no arguments whenever the special key is
    requested, e.g. with ``Giver(special=["$specialkey"])``.

    Use ``sys._getframe(3)`` to get the frame in which give() was called.

    Example:
        .. code-block:: python

            @register_special("$time")
            def _special_time():
                return time.time()

    Arguments:
        key: The key, conventionally starting with a "$".
    """

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


def _find_targets(target):
    if isinstance(target, ast.Tuple):
        results = []
        for t in target.elts:
            results += _find_targets(t)
        return results
    else:
        return [target.id]


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
                        names = _find_targets(target)
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

    * ``len(args) == 0`` => Use the variable assigned in the line before the call.
    * ``len(args) == 1`` => Use the variable the call is assigned to.
    * ``len(args) >= 1`` => Use the variables passed as arguments to the call.

    Arguments:
        frame: The number of frames to go up to find the context.
        func: The Giver object that was called.
        args: The arguments given to the Giver.
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
    """Giver of key/value pairs.

    ``Giver`` is the class of the ``give`` object.

    Arguments:
        keys:
            List of default keys to give. If ``keys=["x"]``, then
            ``self(123)`` will give ``{"x": 123}``.
        special:
            List of special keys to give (e.g. "$line", "$time", etc.)
        extra:
            Extra key/value pairs to give.
        context:
            The ContextVar that contains a list of handlers to call
            when something is given.
        inherited:
            A ContextVar to use for inherited key/value pairs to give,
            as set by ``with self.inherit(key=value): ...``.
        transform:
            A function from dict to dict that modifies the values to
            give.
    """

    def __init__(
        self,
        *,
        keys=None,
        special=[],
        extra={},
        context=global_context,
        inherited=global_inherited,
        transform=None,
    ):
        self.keys = keys
        self.special = special
        self.extra = extra
        self.context = context
        self.inherited = inherited
        self.transform = transform

    def copy(
        self,
        keys=None,
        special=None,
        extra=None,
        context=None,
        inherited=None,
        transform=None,
    ):
        """Copy this Giver with modified parameters."""
        return type(self)(
            keys=self.keys if keys is None else keys,
            special=self.special if special is None else special,
            extra=self.extra if extra is None else extra,
            context=self.context if context is None else context,
            inherited=self.inherited if inherited is None else inherited,
            transform=self.transform if transform is None else transform,
        )

    @property
    def line(self):
        """Return a giver that gives the line where it is called."""
        return self.copy(special=(*self.special, "$line"))

    @property
    def time(self):
        """Return a giver that gives the time where it is called."""
        return self.copy(special=(*self.special, "$time"))

    @contextmanager
    def inherit(self, **keys):
        """Create a context manager within which extra values are given.

        .. code-block:: python

            with give.inherit(a=1):
                give(b=2)   # gives {"a": 1, "b": 2}

        Arguments:
            keys: The key/value pairs to give within the block.
        """
        inh = self.inherited.get()
        token = self.inherited.set({**inh, **keys})
        try:
            yield
        finally:
            self.inherited.reset(token)

    @contextmanager
    def wrap(self, name, **keys):
        """Create a context manager that marks the beginning/end of the block.

        ``wrap`` first creates a unique ID to identify the block,
        then gives the ``$wrap`` sentinel with name, uid and step="begin"
        at the beginning of it gives the same ``$wrap`` but with step="end"
        at the end of the block.

        :meth:`giving.gvn.ObservableProxy.wrap` is the corresponding
        method on the ObservableProxy returned by ``given()`` and it
        can be used to wrap another context manager on the same block.
        :meth:`giving.gvn.ObservableProxy.group_wrap` is another method
        that uses the sentinels produced by ``wrap``.

        .. code-block:: python

            with give.wrap("W", x=1):  # gives: {"$wrap": {"name": "W", "step": "begin", "id": ID}, "x": 1}
                ...
            # end block, gives: {"$wrap": {"name": "W", "step": "end", "id": ID}, "x": 1}

        Arguments:
            name: The name to associate to this wrap block.
            keys: Extra key/value pairs to give along with the sentinels.
        """
        num = next(global_count)
        self.produce({"$wrap": {"name": name, "step": "begin", "id": num}, **keys})
        try:
            yield
        finally:
            self.produce({"$wrap": {"name": name, "step": "end", "id": num}, **keys})

    @contextmanager
    def wrap_inherit(self, name, **keys):
        """Shorthand for using wrap and inherit.

        .. code-block:: python

            with give.wrap_inherit("W", a=1):
                ...

        Is equivalent to:

        .. code-block:: python

            with give.inherit(a=1):
                with give.wrap("W"):
                    ...

        Arguments:
            name: The name to associate to this wrap block.
            keys: Key/value pairs to inherit.
        """
        with self.inherit(**keys):
            with self.wrap(name):
                yield

    def produce(self, values):
        """Give the values dictionary."""
        for special in self.special:
            values[special] = special_keys[special]()

        if self.extra:
            values = {**self.extra, **values}

        inh = self.inherited.get()
        if inh is not None:
            values = {**inh, **values}

        for handler in self.context.get():
            handler(values)

    def variant(self, fn):
        """Create a version of give that transforms the data.

        .. code-block:: python

            @give.variant
            def give_image(data):
                return {"image": data}

            ...

            give_image(x, y)  # gives {"image": {"x": x, "y": y}}

        Arguments:
            fn: A function from a dict to a dict.
            give: The base give function to wrap (defaults to global give).
        """
        return self.copy(transform=fn)

    def __call__(self, *args, **values):
        """Give the args and values."""
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

            if self.transform:
                values = self.transform(values)

            self.produce(values)

        if len(args) == 1:
            return args[0]
        else:
            return None


def giver(*keys, **extra):
    """Create a Giver to give the specified keys, plus extra values.

    .. code-block:: python

        g = giver("x", y=1)
        give(3)   # gives {"x": 3, "y": 1}

    """
    normal = [k for k in keys if not k.startswith("$")]
    special = [k for k in keys if k.startswith("$")]
    return Giver(keys=normal, special=special, extra=extra)
