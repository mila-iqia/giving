
Integrations
============

In this page, we will see how to integrate a selection of existing logging services or display/plotting libraries with Giving. This list is **not exhaustive**: in fact, there is really nothing specific to do to integrate Giving with these libraries.

The main advantage of Giving is *decoupling*: calls to ``give`` in your code produce a stream of raw data, and then that stream can be manipulated in powerful ways to create advanced displays, to forward data to an external server, and so on.

Using ``give`` uniformly for logging data also allows deferring the decision of what to log where. For instance, you might want to log some piece of data continuously, some other piece of data every 30 seconds, and you might want to send another into a web service. And you might want to change these decisions later on.


Rich
----

Rich_ is a Python library for rich, colourful and interactive terminal applications. It supports representations of Python objects, tables, progress bars, and so on, so it is a prime integration target.

.. _Rich: https://github.com/willmcgugan/rich

TODO


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
