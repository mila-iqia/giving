
Motivation
==========

Giving implements what we may call **functional reactive logging**. The basic idea is that you can log objects anywhere in your program, as easily as you can print them. These objects are then aggregated by Giving into **Observable streams of dictionaries** with which you can do pretty much :ref:`anything you want<OperatorList>`, including printing them.

The main advantage of Giving is *decoupling*: calls to ``give`` in your code produce a stream of raw data, and then that stream can be manipulated in powerful ways to create advanced displays, to forward data to an external server, and so on.

Using ``give`` uniformly for logging data also allows deferring the decision of what to log where. For instance, you might want to log some piece of data continuously, some other piece of data every 30 seconds, and you might want to send another into a web service. And you might want to change these decisions later on.


Getting started
===============

Install
-------

You can trivially install ``giving`` via ``pip``:

.. code-block:: bash

    pip install giving


Usage
-----

Giving provides two main functions, :func:`~giving.core.give` and :func:`~giving.core.given`.

* **give** is used to log data into a stream. For example, ``give(a=1, b=2)`` will push ``{"a": 1, "b": 2}`` onto the stream.
* **given** is a context manager that gives access to that stream. Using methods such as :func:`~giving.operators.where`, :func:`~giving.operators.map` or :func:`~giving.operators.filter`, the stream can be transformed.

Here is a simple example. First, add calls to ``give`` into the main code:

.. code-block:: python

    from giving import give

    def bisect(arr, key):
        lo = -1
        hi = len(arr)
        while lo < hi - 1:
            mid = lo + (hi - lo) // 2
            give(mid)                # push {"mid": mid}
            if arr[mid] > key:
                hi = mid
                give()               # push {"hi": hi}
            else:
                lo = mid
                give()               # push {"lo": lo}
        return lo + 1

Second, wrap the invocation in a ``given`` block:

.. code-block:: python

    from giving import given

    with given() as gv:    # gv represents the stream
        # We will print everything that is given as-is
        gv.print()

        # The function must be run *after* we set up our pipeline
        bisect(list(range(10)), 3)

When the above is run, the following is printed:

.. code-block:: python

    {'mid': 4}
    {'hi': 4}
    {'mid': 1}
    {'lo': 1}
    {'mid': 2}
    {'lo': 2}
    {'mid': 3}
    {'lo': 3}

This is simple enough (and useful enough), but the fun starts when you use ``gv``'s many methods:

.. code-block:: python

    with given() as gv:
        # gv["?mid"] is equivalent to gv.where("mid")["mid"], it extracts mid and ignores the rest
        # min() outputs the minimum at the end of the stream
        gv["?mid"].min().print("min(mid): {}")

        # kscan() incrementally merges dictionaries in the stream with the previous ones
        # The first few outputs of kscan() will not have all 3 values, hence skip_missing
        gv.kscan().print("{lo} <= {mid} <= {hi}", skip_missing=True)

        # Trigger a breakpoint whenever lo > hi
        gv.kscan().where("lo", "hi").kfilter(lambda lo, hi: lo > hi).breakpoint()

        # Put the values of mid in an array
        mids = gv["?mid"].accum()

        bisect(list(range(10)), 3)

    print(mids)

The above will output this:

.. code-block::

    1 <= 1 <= 4
    1 <= 2 <= 4
    2 <= 2 <= 4
    2 <= 3 <= 4
    3 <= 3 <= 4
    min(mid): 1
    [4, 1, 2, 3]
