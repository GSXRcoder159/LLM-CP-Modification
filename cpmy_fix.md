# A bug in the `cpmpy` library

There is a bug in CPMpy's normalisation pass (the `real_to_int`) function in `cpmpy/transformations/normalize.py` that internally performs:

```python
expr.args = reel_to_int(expr.args)
```

And since every CPMpy `Expression` has its operands stored in a private list `_args` and exposes them via a read-only `args` property, assigning to `expr.args` always raises:
> AttributeError: Cannot modify read-only attribute 'args', use ‘update_args()’

## The fix

To fix this the line in `cpmpy/transformations/normalize.py` needs to be replaced with:

```python
expr.update_args(reel_to_int(expr.args))
```

which uses the proper API for in-place argument updates.
