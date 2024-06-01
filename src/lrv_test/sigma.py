from typing import Callable

import numpy as np
from scipy.integrate import quad

from lrv_test.contour import Contour
from lrv_test.functions import support_MP, z_t_t_tilde
from lrv_test.types import real_function


def s(z: complex, c: float) -> complex:
    z_t_t_tilde_square = z_t_t_tilde(z, c) ** 2
    return np.sqrt(c) * z_t_t_tilde_square / (1 - c * z_t_t_tilde_square)


def _contour_integral(integrand: Callable, contour: Contour):
    integrand_reparametrized = lambda t: integrand(contour.z(t)) * contour.dz(t)
    return (
        quad(
            lambda z: np.real(integrand_reparametrized(z)),
            0,
            1,
            epsabs=1e-2,  # integral_config.epsabs,
            epsrel=1e-6,  # integral_config.epsrel,
        )[0]
        + 1j
        * quad(
            lambda z: np.imag(integrand_reparametrized(z)),
            0,
            1,
            epsabs=1e-2,  # integral_config.epsabs,
            epsrel=1e-6,  # integral_config.epsrel,
        )[0]
    )


def compute_sigma(f: real_function, c: float) -> float:
    # create the contour used in contour integration
    support = support_MP(c)
    radius = (support[1] - support[0]) / 2 +1
    center = (support[0] + support[1]) / 2
    contour = Contour.from_circle_parameters(center, radius)

    # the integral is separable!
    contour_integral = lambda n: _contour_integral(
        lambda z: f(z) * s(z, c) * (np.sqrt(c) * z_t_t_tilde(z, c)) ** n, contour
    )

    N = 100 
    cis = np.array([contour_integral(n) for n in range(N)])
    result = np.sqrt(c * np.sum([(n + 1) * ci**2 for n, ci in enumerate(cis)]) / (4*np.pi**2))
    assert np.imag(result) < 1e-1
    return result 
