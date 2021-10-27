
# giving — the reactive logger

[Documentation](https://giving.readthedocs.io)

`giving` is a simple, magical library that lets you log or "give" arbitrary data throughout a program and then process it as an event stream. You can use it to log to the terminal, to [wandb](https://wandb.ai/site) or [mlflow](https://mlflow.org/), to compute minimums, maximums, rolling means, etc., separate from your program's core logic.

1. Inside your code, **`give()`** every object or datum that you may want to log or compute metrics about.
2. Wrap your main loop with **`given()`** and define pipelines to map, filter and reduce the data you gave.


## Examples


<table>
<tr>
<th>Code</th>
<th>Output</th>
</tr>

<!-- ROW -->

<tr>
<td>

Simple logging

```python
# All calls to give() will log to the configured console
with given().display():
    a, b = 10, 20
    # Without parameters: last expression + result
    give()
    # With parameters:
    # parameter is just value: value => value
    # parameter is key and value: key => value
    give(a * b, c=30)
```

</td>
<td>

```
a: 10; b: 20
a * b: 200; c: 30
```

</td>
</tr>
<tr></tr>

<!-- ROW -->

<tr>
<td>

Extract values into a list

```python
# give(key=value) with key == "s" will add value to `results`
with given()["s"].values() as results:
    s = 0
    for i in range(5):
        s += i
        give(s)

print(results)
```

</td>
<td>

```
[0, 1, 3, 6, 10]
```

</td>
</tr>
<tr></tr>

<!-- ROW -->

<tr>
<td>

Reductions (min, max, count, etc.)

```python
def collatz(n):
    while n != 1:
        give(n)
        n = (3 * n + 1) if n % 2 else (n // 2)

with given() as gv:
    gv["n"].max().print("max: {}")
    gv["n"].count().print("steps: {}")

    collatz(2021)
```

</td>
<td>

```
max: 6064
steps: 63
```

</td>
</tr>
<tr></tr>

<!-- ROW -->

<tr>
<td>

Using the `eval` method instead of `with`:

```python
st, = given()["n"].count().eval(collatz, 2021)
print(st)
```

</td>
<td>

```
63
```

</td>
</tr>
<tr></tr>

<!-- ROW -->

<tr>
<td>

The `kscan` method

```python
with given() as gv:
    gv.kscan().display()

    give(elk=1)
    give(rabbit=2)
    give(elk=3, wolf=4)
```

</td>
<td>

```
elk: 1
elk: 1; rabbit: 2
elk: 3; rabbit: 2; wolf: 4
```

</td>
</tr>
<tr></tr>

<!-- ROW -->

<tr>
<td>

The `throttle` method

```python
with given() as gv:
    gv.throttle(1).display()

    for i in range(50):
        give(i)
        time.sleep(0.1)
```

</td>
<td>

```
i: 0
i: 10
i: 20
i: 30
i: 40
```

</td>
</tr>
<tr></tr>

</table>

The above examples only show a small number of [all the available operators](https://giving.readthedocs.io/en/latest/ref-operators.html).


## Give

There are multiple ways you can use `give`. `give` returns None *unless* it is given a single positional argument, in which case it returns the value of that argument.

* **give(key=value)**

  This is the most straightforward way to use `give`: you write out both the key and the value associated.

  *Returns:* None

* **x = give(value)**

  When no key is given, but the result of `give` is assigned to a variable, the key is the name of that variable. In other words, the above is equivalent to `give(x=value)`.

  *Returns:* The value

* **give(x)**

  When no key is given and the result is *not* assigned to a variable, `give(x)` is equivalent to `give(x=x)`. If the argument is an expression like `x * x`, the key will be the string `"x * x"`.

  *Returns:* The value

* **give(x, y, z)**

  Multiple arguments can be given. The above is equivalent to `give(x=x, y=y, z=z)`.

  *Returns:* None

* **x = value; give()**

  If `give` has no arguments at all, it will look at the immediately previous statement and infer what you mean. The above is equivalent to `x = value; give(x=value)`.

  *Returns:* None


## Important functions and methods

* [print](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.print) and [display](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.display): for printing out the stream
* [values](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.values) and [accum](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.accum): for accumulating into a list
* [subscribe](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.subscribe) and [ksubscribe](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.ksubscribe): perform a task on every element
* [where](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.where), [where_any](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.where_any), [keep](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.keep), `gv["key"]`, `gv["?key"]`: filter based on keys

[See here for more details.](https://giving.readthedocs.io/en/latest/guide.html#important-methods)


## Operator summary

Not all operators are listed here. [See here](https://giving.readthedocs.io/en/latest/ref-operators.html) for the complete list.

### Filtering

* [filter](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.filter): filter with a function
* [kfilter](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.kfilter): filter with a function (keyword arguments)
* [where](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.where): filter based on keys and simple conditions
* [where_any](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.where_any): filter based on keys
* [keep](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.keep): filter based on keys (+drop the rest)
* [distinct](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.distinct): only emit distinct elements
* [norepeat](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.norepeat): only emit distinct consecutive elements
* [first](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.first): only emit the first element
* [last](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.last): only emit the last element
* [take](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.take): only emit the first n elements
* [take_last](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.take_last): only emit the last n elements
* [skip](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.skip): suppress the first n elements
* [skip_last](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.skip_last): suppress the last n elements

### Mapping

* [map](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.map): map with a function
* [kmap](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.kmap): map with a function (keyword arguments)
* [augment](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.augment): add extra keys using a mapping function
* [getitem](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.getitem): extract value for a specific key
* [sole](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.sole): extract value from dict of length 1
* [as_](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.as_): wrap as a dict

### Reduction

* [reduce](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.reduce): reduce with a function
* [scan](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.scan): emit a result at each reduction step
* [roll](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.roll): reduce using overlapping windows
* [kmerge](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.kmerge): merge all dictionaries in the stream
* [kscan](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.kscan): incremental version of ``kmerge``

### Arithmetic reductions

Most of these reductions can be called with the ``scan`` argument set to ``True`` to use ``scan`` instead of ``reduce``. ``scan`` can also be set to an integer, in which case ``roll`` is used.

* [average](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.average)
* [average_and_variance](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.average_and_variance)
* [count](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.count)
* [max](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.max)
* [min](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.min)
* [sum](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.sum)
* [variance](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.variance)

### Wrapping

* [wrap](https://giving.readthedocs.io/en/latest/ref-gvr.html#giving.gvr.Giver.wrap): give a special key at the beginning and end of a block
* [wrap_inherit](https://giving.readthedocs.io/en/latest/ref-gvr.html#giving.gvr.Giver.wrap_inherit): give a special key at the beginning and end of a block
* [inherit](https://giving.readthedocs.io/en/latest/ref-gvr.html#giving.gvr.Giver.inherit): add default key/values for every give() in the block
* [wrap](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.wrap): plug a context manager at the location of a ``give.wrap``
* [kwrap](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.kwrap): same as wrap, but pass kwargs

### Timing

* [debounce](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.debounce): suppress events that are too close in time
* [sample](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.sample): sample an element every n seconds
* [throttle](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.throttle): emit at most once every n seconds

### Debugging

* [breakpoint](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.breakpoint): set a breakpoint whenever data comes in. Use this with filters.
* [tag](https://giving.readthedocs.io/en/latest/ref-operators.html#giving.operators.tag): assigns a special word to every entry. Use with ``breakword``.
* [breakword](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.breakword): set a breakpoint on a specific word set by ``tag``, using the ``BREAKWORD`` environment variable.

### Other

* [accum](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.accum): accumulate into a list
* [display](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.display): print out the stream (pretty).
* [print](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.print): print out the stream.
* [values](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.values): accumulate into a list (context manager)
* [subscribe](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.subscribe): run a task on every element
* [ksubscribe](https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given.ksubscribe): run a task on every element (keyword arguments)


## ML ideas

Here are some ideas for using giving in a machine learning model training context:

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
        # >> is shorthand for .subscribe()
        losses >> wandb.log

        # Print the minimum loss at the end
        losses["loss"].min().print("Minimum loss: {}")

        # Print the mean of the last 100 losses
        # * affix adds columns, so we will display i, loss and meanloss together
        # * The scan argument outputs the mean incrementally
        # * It's important that each affixed column has the same length as
        #   the losses stream (or "table")
        losses.affix(meanloss=losses["loss"].mean(scan=100)).display()

        # Store all the losses in a list
        losslist = losses["loss"].accum()

        # Set a breakpoint whenever the loss is nan or infinite
        losses["loss"].filter(lambda loss: not math.isfinite(loss)).breakpoint()


        # Filter all the lines that have the "model" key:
        models = gv.where("model")

        # Write a checkpoint of the model at most once every 30 minutes
        models["model"].throttle(30 * 60).subscribe(
            lambda model: model.checkpoint()
        )

        # Watch with wandb, but only once at the very beginning
        models["model"].first() >> wandb.watch

        # Write the final model (you could also use models.last())
        models.where(final=True)["model"].subscribe(
            lambda model: model.save()
        )


        # ===========================================================
        # Finally, execute the code. All the pipelines we defined above
        # will proceed as we give data.
        # ===========================================================
        main()
```
