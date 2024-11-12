import mlx.core as mx
import numpy as np
import pytest
from lrv_test.contour import Contour
from lrv_test.functions import support_MP, t
from lrv_test.utils import action_D_on_f, contour_integral, derivative


@pytest.mark.parametrize(
    "f, x, expected",
    [
        (lambda x: x**2, 2, 4),
        (lambda x: x**3, 2, 12),
    ],
)
def test_derivative(f, x, expected):
    assert derivative(f)(x) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "f, c, expected",
    [
        (lambda x: x, 1 / 2, 1),
        (lambda x: x**2, 1 / 2, 1 / 2),
    ],
)
def p(f, c, expected):
    assert action_D_on_f(f, lambda z: t(z, c), support_MP(c)) == expected


@pytest.mark.parametrize(
    "integrand, center, radius, expected",
    [
        (lambda z: z, 0, 1, 0),
        (lambda z: z**2, 0, 1, 0),
        (lambda z: 1 / z, 0, 1, -2 * np.pi * 1j),
    ],
)
def test_contour_integral(integrand, center, radius, expected):
    contour = Contour.from_circle_parameters(center, radius)
    assert contour_integral(integrand, contour) == pytest.approx(expected, rel=1e-3)
