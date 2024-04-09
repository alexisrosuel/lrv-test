from typing import Optional

import numpy as np
from spectral_coherence import coherence, lag_window

from lrv_test.config import LRVConfig
from lrv_test.known_functions import X_MINUS_1_SQUARE
from lrv_test.lag_window import lag_window
from lrv_test.result import LRVResult
from lrv_test.types import f64_2d, real_function
from lrv_test.utils import derivative


def _v_n(B: int, N: int) -> float:
    b_range = np.arange(-(B - 1) // 2, (B - 1) // 2 + 1)
    return np.mean((b_range / N) ** 2)


def _r_n(sd: real_function, sd_prime: real_function) -> real_function:
    """all the input and output functions on the frequency"""
    values = lambda freq: sd_prime(freq) / sd(freq)
    return lambda freq: np.mean(values(freq)) ** 2


def LRV(
    y: f64_2d, lrv_config: LRVConfig, sd: Optional[real_function] = None
) -> LRVResult:
    """
    Compute the LRV statistics on the time series y. The LRVResult object also
    contains additional details about the statistics computation.
    """
    N, M = y.shape

    # estimate the spectral coherence from the sample
    C_hats, freqs = coherence(y, lrv_config.B, n_max_freqs=lrv_config.n_max_freqs)
    # n_freqs may be less than n_samples if n_max_freqs is not None and n_max_freqs < n_samples
    n_freqs = len(freqs )  

    # for each C_hat, compute the Linear Spectral Statistics associated with f
    eigenvalues = [np.linalg.eigvalsh(C_hat) for C_hat in C_hats]
    LSSs = np.array([np.mean(lrv_config.f(eigenvalues[n])) for n in range(n_freqs)])

    # compute the first corrective terms: v_n 
    v_n = _v_n(lrv_config.B, N)

    # compute the second corrective terms: r_n
    if sd is None:
        sd = lag_window(y, lrv_config.L)
    sd_prime = derivative(sd)
    r_n = _r_n(sd, sd_prime)
    r_n_values = np.array([r_n(freq) for freq in freqs])
   
    # compute the other required terms
    c_n = M / lrv_config.B
    if True or lrv_config.f.__code__.co_code == X_MINUS_1_SQUARE:
        f_mp = c_n
        f_D_n = c_n
        sigma_n = 2 * np.sqrt(2 * c_n**2)
    else:
        # need to compute numerically the integral
        raise NotImplementedError

    return LRVResult(
        N,
        M,
        lrv_config,
        freqs,
        LSSs,
        c_n,
        f_mp,
        f_D_n,
        r_n_values,
        v_n,
        sigma_n,
    )
