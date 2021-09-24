
Examples
========

The examples_ directory contains a few examples that you can try:

.. _examples: https://github.com/breuleux/giving/tree/master/examples

.. code-block:: bash

    git clone git@github.com:breuleux/giving.git
    cd examples/live_bisect
    pip install -r requirements.txt
    python main.py


MNIST Example
-------------

The most elaborate example is the `MNIST example`_ which is essentially a copy of PyTorch's `basic MNIST example`_, adapted to use ``give`` and which demonstrates various ways to log data to the terminal and to logging services such as Weights and Biases.

`View the diff`_.

.. _MNIST example: https://github.com/breuleux/giving/blob/master/examples/mnist/main.py
.. _basic MNIST example: https://github.com/pytorch/examples/tree/master/mnist
.. _View the diff: https://github.com/breuleux/giving/compare/mnist-original...mnist-new


Integrations
============

Giving can be easily integrated with existing logging services or display/plotting libraries. Here are some examples:


Rich
----

Rich_ is a Python library for rich, colourful and interactive terminal applications. It supports representations of Python objects, tables, progress bars, and so on.

.. _Rich: https://github.com/willmcgugan/rich

Consider the following code:

.. code-block:: python

    def bisect(arr, key):
        lo = -1
        hi = len(arr)
        give(lo, hi, mid=None)       # push {"lo": lo, "hi": hi, "mid": None}
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

    def main():
        bisect(list(range(1000)), 742)

    if __name__ == "__main__":
        main()

What we want to do is have a trivial little visualization where we see the values of the ``mid``, ``hi`` and ``lo`` variables.

Here is how to do it:

.. code-block:: python

    import time
    from giving import give, given
    from rich.live import Live
    from rich.table import Table
    from rich.pretty import Pretty

    def dict_to_table(d):
        table = Table.grid(padding=(0, 3, 0, 0))
        table.add_column("key", style="bold green")
        for k, v in d.items():
            if not k.startswith("$"):
                table.add_row(k, Pretty(v))
        return table

    ...

    if __name__ == "__main__":
        with given() as gv:
            live = Live(refresh_per_second=4)
            gv.wrap("main", live)

            @gv.kscan().subscribe
            def _(data):
                time.sleep(0.5)  # Wait a bit so that we can see the values change
                live.update(dict_to_table(data))

            with give.wrap("main"):
                main()

Running the code, you get something like this:

.. image:: https://github.com/breuleux/giving/raw/media/media/bisect.gif


Weights and biases
------------------

`Weights and biases <https://wandb.ai/site>`_ is a popular framework to run machine learning experiments and log various metrics. Normally you would pepper your code with references to ``wandb.log``, but using ``giving``, it is possible to decouple it from your code, thereby reducing your reliance to the service.


Logging metrics
^^^^^^^^^^^^^^^

The ``wandb.log(metrics)`` method takes a dictionary and logs the content. Fortunately, ``give`` produces dictionaries. Therefore, provided you ``give`` all your metrics, you can do something as simple as:

.. code-block:: python

    with given() as gv:
        gv >> wandb.log

        main()

This may log a lot more things than you want, but it is simple to perform a selection. For example, if you only want to log ``train_loss`` and ``test_loss``:

.. code-block:: python

    gv.keep("train_loss", "test_loss") >> wandb.log

Now, what if you want to log a weights matrix, but only every minute? This is kind of tricky normally, but with ``giving`` nothing could be simpler:

.. code-block:: python

    gv.keep("weights").throttle(60) >> wandb.log


Watching your model
^^^^^^^^^^^^^^^^^^^

One of wandb's best features is the ``watch`` method, which will automatically periodically log your parameters. But if we ``give(model)`` at any point, we can extract the very first occurrence (because we only want to call ``watch`` once) and forward it to ``watch``:

.. code-block:: python

    gv["?model"].first() >> wandb.watch


.. note::
    ``gv["?model"]`` is equivalent to ``gv.keep("model")["model"]``.


CometML
-------

`CometML <https://www.comet.ml/>`_ is another logging service, with a slightly different interface. It is, however, still simple to use Giving to log into it.


Wrapping train/test
^^^^^^^^^^^^^^^^^^^

CometML uses context managers to wrap the train and test phases:

.. code-block:: python

    with experiment.train():
        ...

    with experiment.test():
        ...

Instead, you can use ``give.wrap``:

.. code-block:: python

    with give.wrap("train"):
        ...

    with give.wrap("test"):
        ...

    ...

    with given() as gv:
        gv.wrap("train", experiment.train)
        gv.wrap("test", experiment.test)

At a first glance that is just some extra code, but:

1. The ``experiment`` object does not need to be instantiated or passed to train/test.
2. The dependency to ``comet_ml`` can therefore be isolated to the ``given`` block.
3. Multiple context managers can be plugged into the same ``give.wrap`` block, for example a progress bar.

Logging
^^^^^^^

CometML offers a whole host of logging methods, so it is a little less straightforward than WandB, but it is nonetheless similar. For example, to specifically log ``loss``:

.. code-block:: python

    # Optionally set epoch/step whenever epoch or step is given
    gv["?epoch"] >> experiment.set_epoch
    gv["?step"] >> experiment.set_step

    # Log the loss.
    # The train/test wrappers will infer whether it's train or test loss
    gv.keep("loss") >> experiment.log_metrics

If you want to log an image every 60 seconds (given as ``give(image=...)``):

.. code-block:: python

    gv["?image"].throttle(60) >> experiment.log_image
