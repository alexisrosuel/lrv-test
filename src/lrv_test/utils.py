from typing import Callable

import numpy as np
from scipy.integrate import quad

from lrv_test.contour import Contour
from lrv_test.types import real_function


def derivative(f: real_function) -> real_function:
    epsilon = 1e-6
    return lambda x: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


def action_D_on_f(
    f: real_function,
    g: Callable[[complex], complex],
    support: tuple[float, float],
    tolerance: float,
) -> float:
    # Stietljes inversion formula to get the density from the transform, integrated
    # against f
    y = 1e-20
    low, high = support
    value, error = quad(lambda x: f(x) * np.imag(g(x + 1j * y)) / np.pi, low, high)
    assert error < tolerance
    return value


def contour_integral(integrand: Callable, contour: Contour) -> complex:
    integrand_reparametrized = lambda t: integrand(contour.z(t)) * contour.dz(t)
    return quad(
        lambda z: integrand_reparametrized(z),
        0,
        1,
        epsabs=1e-6,
        epsrel=1e-6,
        complex_func=True,
    )[0]


def psi(w: float, c: float) -> float:
    return (w + 1) * (w + c) / w
