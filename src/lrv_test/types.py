from typing import Callable, NewType

import numpy as np

f64_1d = NewType("f64_1d", np.ndarray)
f64_2d = NewType("f64_2d", np.ndarray)
complex_1d = NewType("complex_1d", np.ndarray)
complex_2d = NewType("complex_2d", np.ndarray)

real_function = tuple[Callable[[float], float]]
