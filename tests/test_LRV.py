import numpy as np
import pytest
from clt_lss_coherence.LRV import LRV, LRVConfig, LRVResult, _r_n, _v_n
from clt_lss_coherence.utils import X_MINUS_1_SQUARE


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


@pytest.mark.parametrize(
    "lrv_config, expected_lrv_result",
    [
        (
            LRVConfig(B=3, L=2, f=lambda x: (x - 1) ** 2, n_max_freqs=3),
            LRVResult(
                N=10,
                M=2,
                lrv_config=LRVConfig(B=3, L=2, f=lambda x: (x - 1) ** 2, n_max_freqs=3),
                freqs=np.array([-0.5, -0.1, 0.4]),
                lsss=np.array([1.0, 0.99875056, 1.0]),
                c_n=0.6666666666666666,
                f_mp=0.6666666666666666,
                f_Dn=0.6666666666666666,
                r_n_hat=np.array(
                    [
                        0.00000000e00 + 0.00000000e00j,
                        8.21937666e01 + 2.13479529e-10j,
                        7.49408623e05 + 0.00000000e00j,
                    ]
                ),
                v_n=0.006666666666666668,
                sigma_N=1.8856180831641267,
                r_n=None,
            ),
        ),
    ],
)
def test_LRV(lrv_config, expected_lrv_result):
    y = np.arange(10 * 2).reshape(10, 2)
    assert LRV(y, lrv_config) == expected_lrv_result
