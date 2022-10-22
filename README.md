# Simple Autograd
A simple autograd implementation based on NumPy's 
`ndarrays` with a PyTorch-like API.
Hence, it is also based on *dynamic* computation graphs.

To enable graph computation, simply wrap any NumPy array 
with `Variable` (a subclass of `np.ndarray`):

```python
import numpy as np
from simple_autograd import Variable

a = np.random.randn(3)
v = Variable(a)
(v * 3).sum().backward()
v.grad  # array([3., 3., 3.])
```

Although simple, the engine is able to handle a wide variety
of operations. For example, the following relatively complex
expression calculates the correct gradient for `x` 
(checked against [equivalent](test/test_variable.py) in PyTorch):

```python
x = Variable(np.random.randn(3, 4, 5))
y = np.random.randn(5, 6)

(
    ((x @ y).cos().mean((0,-1), keepdim=True)[:,[1,1,2,-1,0,0],None,:] * y)
    .swapaxes(0, 1)
    .var()
    .backward()
)
x.grad  # same (up to precision errors) as torch
```

## Included Batteries
The engine also includes a tiny neural network library under the submodule
`nn`.
This contains among others:
- Convolution operator
- Layers like `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d` etc.
- Optimizers (`SGD` and `Adam`)
- Dataset loading (`MNIST`)

For a comprehensive example using these, please see the included
[training script](examples/train.py) under [examples](examples).

The purpose of this was to test and demonstrate the capabilities of the engine.
The script can train a simple MLP using `nn` up to +95% test accuracy on the MNIST
dataset within a few minutes.
(To be fair, this is probably over-engineered for that. 
However, that was also the point - to come up with an API as modular and simple to use as PyTorch's.)

For more details, see corresponding README under [examples](examples).




## Installation
Although, `torch` is used for testing,
the only real dependency is `numpy` for 
the engine itself.

The included `nn` submodule contains a convolution
operator, which requires `scipy`.
However, it is optional.

`setup.py` intentionally does not install any dependencies.
As such, you have to install them manually if you are using a fresh environment.
- Just the autograd engine
  ```sh
  pip install numpy
  ```
- Convolution as well
  ```sh
  pip install numpy scipy
  ```
- Testing against `torch`
  ```sh
  pip install numpy scipy torch
  # it's probably more sensible to use an existing environment containing torch
  ```

`tqdm` is also optional but recommended when using the provided 
[training script](examples/train.py).


To install repo, you can use:
```sh
pip install git+https://github.com/nps1ngh/simple_autograd
```

## Why?
The primary aim of this project was to serve as a learning exercise to understand
computational graph engineering better.
I wanted to try to recreate the PyTorch API based on my experience using the API
and the theoretical knowledge behind computational graphs from my deep learning 
classes at university.
I think it has turned out quite well, and I'm quite satisfied with it.
