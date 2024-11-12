import numpy as np
from scipy.integrate import quad

from lrv_test.contour import Contour
from lrv_test.functions import z_t_t_tilde
from lrv_test.types import real_function
from lrv_test.utils import contour_integral, psi


def s(z: complex, c: float) -> complex:
    z_t_t_tilde_square = z_t_t_tilde(z, c) ** 2
    return np.sqrt(c) * z_t_t_tilde_square / (1 - c * z_t_t_tilde_square)


def rectangle_integral(
    f: real_function, min_x: float, max_x: float, min_y: float, max_y: float
) -> float:
    # compute the four integrals
    integral_1 = quad(lambda x: f(x + 1j * max_y), min_x, max_x, complex_func=True)[0]
    integral_2 = quad(lambda y: f(max_x + 1j * y), max_y, min_y, complex_func=True)[0]
    integral_3 = quad(lambda x: f(x + 1j * min_y), max_x, min_x, complex_func=True)[0]
    integral_4 = quad(lambda y: f(min_x + 1j * y), min_y, max_y, complex_func=True)[0]

    # return the sum of the four integrals
    return integral_1 + integral_2 + integral_3 + integral_4


def compute_sigma(f: real_function, c: float, tolerance: float) -> float:
    """
    Very fast computation of the limiting variance of the test statistic under H0.
    This value is initially a double contour integral, which is slow to compute.
    However, it can be proven that using an adequate change of variable, this double
    integral is equal to the infinite (converging) sum of square of a single contour
    integral. This single contour integral is very fast to compute, and the sum
    converges sufficiently fast for our needs.
    """
    radius = np.sqrt(c) + 0.1
    center = 0
    contour = Contour.from_circle_parameters(center, radius)

    def integrand(w, n):
        return -np.sqrt(n + 1) * c * f(psi(w, c)) / w ** (n + 2)

    ci = lambda n: contour_integral(
        integrand=lambda w: integrand(w, n), contour=contour
    )

    # compute the integrals until a term is small enough
    cis = []
    n = 1
    integral_value = ci(n)
    while np.abs(integral_value) > 1e-4 and n < 20:
        cis.append(integral_value)
        n += 1
        integral_value = ci(n)
    cis.append(integral_value)

    result = -np.sum([ci**2 for ci in cis]) / (4 * np.pi**2)
    assert np.imag(result) < tolerance

    return np.sqrt(np.real(result))
