
# giving -- the reactive logger

`giving` is a simple, magical library that lets you log or "give" arbitrary data throughout a program and then process it as an event stream.

1. Inside your code, call **`give`** on every object or datum that you may want to include in your metrics.
2. Wrap your main loop with **`given`** and define pipelines to map, filter and reduce the data you gave.

Here are tasks that are made easier and cleaner by `giving`:

* Log some data periodically (e.g. save a model checkpoint every 30 minutes)
* Compute the rolling mean of some data in a loop (or min, max, etc.)
* Set breakpoints on complex conditions
* Isolate all logging logic (e.g. you want to use [wandb](https://wandb.ai/site) or [mlflow](https://mlflow.org/) but want neither of them in your core logic)


Here is an example:


```python
from giving import give, given


def main():
    model = Model()

    for i in range(niters):
        # Give the model. give looks at the argument string, so 
        # give(model) is equivalent to give(model=model)
        give(model)

        loss = model.step()

        # Give the iteration number and the loss (equivalent to give(i=i, loss=loss))
        give(i, loss)

    # Give the final model. The final=True key is there so we can filter on it.
    give(model, final=True)


if __name__ == "__main__":
    # Now here are ideas of what you can do with what was given

    with given() as gv:
        # ===========================================================
        # Define our pipeline **before** running main()
        # ===========================================================

        # Filter all the lines that have the "loss" key
        # NOTE: Same as gv.filter(lambda values: "loss" in values)
        losses = gv.where("loss")

        # Print the losses on stdout
        losses.display()                 # always
        losses.throttle(1).display()     # OR: once every second
        losses.slice(step=10).display()  # OR: every 10th loss

        # Log the losses (and indexes i) with wandb
        losses.subscribe(wandb.log)

        # Print the minimum loss at the end
        losses["loss"].min().format("Minimum loss: {}").display()

        # Print the mean of the last 100 losses
        # * affix adds columns, so we will display i, loss and meanloss together
        # * The scan argument outputs the mean incrementally
        # * It's important that each affixed column has the same length as
        #   the losses stream (or "table")
        losses.affix(
            meanloss=losses["loss"].mean(scan=100)
        ).display()

        # Store all the losses in a list
        losslist = []
        losses["loss"] >> losslist
        # NOTE: The above is shorthand for losses["loss"].subscribe(losslist.append)

        # Set a breakpoint whenever the loss is nan or infinite
        losses["loss"].filter(lambda loss: not math.isfinite(loss)).breakpoint()


        # Filter all the lines that have the "model" key:
        models = gv.where("model")

        # Write a checkpoint of the model at most once every 30 minutes
        models["model"].throttle(30 * 60).subscribe(
            lambda model: model.checkpoint()
        )

        # Watch with wandb, but only once at the very beginning
        models["model"].first().subscribe(wandb.watch)

        # Write the final model (you could also use models.last())
        models.where(final=True)["model"].subscribe(
            lambda model: model.save()
        )


        # ===========================================================
        # Finally, execute the code. All the pipelines we defined above
        # will proceed as we give data.
        # ===========================================================
        main()


    # ===========================================================
    # Extra tasks using extracted data
    # ===========================================================

    # The losslist has been populated, so we could use it to make a plot,
    # or whatever else you want, it's a list :)
    plot(x=range(len(losslist)), y=losslist)
```


## Usage



# Operators


## affix

The `affix` operator allows you to augment a stream of dictionaries with another key, typically computed from the others. The affixed streams must have the same length as the main one, so reductions should be done using `scan` or the `scan` argument. For example:

```python
with given() as g:

    # Only keep entries that have both keys "x" and "y"
    data = g.where("x", "y")

    # Affix various quantities
    data = data.affix(
        # x + y
        xpy=data.keymap(lambda x, y: x + y),

        # The current minimums of x and y
        minx=data["x"].min(scan=True),
        miny=data["y"].min(scan=True),

        # Count the number of positive x in the last 100 values of x
        posx=data["x"].count(lambda x: x > 0, scan=100),
    )

    # You can keep affixing more entries
    data = data.affix(
        # The current minimum of x and y
        min=data.keymap(lambda minx, miny: min(minx, miny))
    )

    # Register for display
    data.display()
```

The advantage of `affix` is that it makes it relatively easy to augment a stream for display, and unlike `map` some of the affixed values can be reductions.
