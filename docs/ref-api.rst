
giving.api
==========

.. automodule:: giving.api

    .. function:: give(*values, **keyvalues)

        Give one or more key/value pairs, accessible through ``given``.

        There are many ways to ``give``:

        * ``give(key=value)`` gives a specific key/value pair.
        * ``key = give(value)`` gives the value associated to the variable
          name it is assigned to.
        * ``give(x)`` gives a value. If it is not the right hand side of
          an assignment, key name is the argument string extracted from the
          source code (in this case, "x").
        * ``give()`` must be immediately below an assignment.
        * Different methods can be combined, for example ``give(x, y, z=2)``
          which is equivalent to ``give(x=x, y=y, z=2)``.

        .. code-block:: python

            give(x=1)      # give {"x": 1}
            y = give(x)    # give {"y": x}
            give(y)        # give {"y": 2}
            give(y * y)    # give {"y * y": 4}
            z = y * y
            give()         # give {"z": 4}

        Returns:
            If *exactly one* non-keyword argument is provided, the argument's
            value is returned. Otherwise, ``give`` returns ``None``.

    .. function:: given()

        Creates a :class:`~giving.gvn.Given` that emits everything that is fed into :func:`~giving.api.give`.

        Instances of Given must be activated as a context manager in order to work, and each instance can only be activated once.

        .. code-block:: python

            def do_things():
                ...
                give(x=1)
                ...
                give(x=2)
                ...

            with given() as gv:
                gv["x"].print(f"x = {x}")

                do_things()

    .. autofunction:: make_give
