# Interesting things

Some cool things I found out about:

- `%` has a gradient (with respect to the first operand) 
equal to 1 a.e. (see [forum post](https://discuss.pytorch.org/t/fmod-or-remainder-runtimeerror-the-derivative-for-other-is-not-implemented/64276/5)
and [issue](https://github.com/tensorflow/tensorflow/issues/6365#issuecomment-286229848)).

- When _completely_ reducing using `.min()` or `.max()` a Tensor, then PyTorch distributes the
incoming gradient equally to all min-/maximum elements in the Tensor. However, this is not the case
when you use `.min(0)` or `.max(0)` or some other specified dimension. In this case, the gradient
is only added to the first occurrence. This exception also occurs even if the Tensor is 1 dimensional.

- NumPy's `ufunc.at` functions allow the `ufunc` operation to be done multiple times on a single element
if the index to it is provided multiple times. (See `operations.IndexingBackward`.)
