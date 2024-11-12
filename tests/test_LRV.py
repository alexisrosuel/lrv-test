import numpy as np
import pytest
from lrv_test.LRV import _r_n, _v_n


@pytest.mark.parametrize(
    "B, N, expected",
    [
        (2, 10, 0.005000000000000001),
    ],
)
def test__v_n(B, N, expected):
    assert _v_n(B, N) == expected


@pytest.mark.parametrize(
    "sd, sd_prime, freq, expected",
    [
        (lambda x: np.array([x]), lambda x: np.array([x]), 1, 1),
        (lambda x: np.array([x]), lambda x: np.array([x]), 2, 1),
        (lambda x: np.array([x]), lambda x: np.array([x**2]), 2, 4),
        (lambda x: np.array([x, x]), lambda x: np.array([x, x**2]), 2, 2.25),
    ],
)
def test__r_n(sd, sd_prime, freq, expected):
    assert _r_n(sd, sd_prime)(freq) == expected
