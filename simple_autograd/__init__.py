import warnings

try:
    import simple_autograd.nn as nn
except ImportError as e:
    warnings.warn(
        f"Error importing `nn`! Only importing autograd engine. {e}"
    )

from simple_autograd.variable import Variable
from simple_autograd.backprop import backward
