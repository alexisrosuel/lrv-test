import numpy as np
import pytest
from lrv_test.sigma import compute_sigma


@pytest.mark.parametrize(
    "f, c, tolerance, expected_sigma",
    [
        (lambda x: x**2, 1, 1e-10, np.sqrt(2)),
        (lambda x: x**2, 1 / 2, 1e-10, np.sqrt(2) / 2),
    ],
)
def test_compute_sigma(f, c, tolerance, expected_sigma):
    sigma = compute_sigma(f, c, tolerance)
    assert sigma == pytest.approx(expected_sigma, rel=tolerance)
