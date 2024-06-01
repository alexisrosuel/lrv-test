from typing import Callable

import numpy as np
from scipy.integrate import quad

from lrv_test.types import real_function


def derivative(f: real_function) -> real_function:
    epsilon = 1e-6
    return lambda x: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


def action_D_on_f(
    f: real_function, g: Callable[[complex], complex], support: tuple[float, float]
) -> float:
    # Stietljes inversion formula to get the density from the transform, integrated
    # against f
    y = 1e-10
    low, high = support
    return np.imag(
        quad(
            lambda x: f(x) * g(x + 1j * y) / np.pi,
            low * 0.5,
            high * 2,
            complex_func=True,
        )
    )[0]
