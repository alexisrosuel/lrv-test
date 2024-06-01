from typing import Optional

import numpy as np
from spectral_coherence import coherence, lag_window

from lrv_test.functions import support_MP, t, z_t_t_tilde
from lrv_test.lag_window import lag_window
from lrv_test.types import f64_1d, f64_2d, real_function
from lrv_test.utils import action_D_on_f, derivative


def _v_n(B: int, N: int) -> float:
    b_range = np.arange(-(B - 1) // 2, (B - 1) // 2 + 1)
    return np.mean((b_range / N) ** 2)


def _r_n(sd: real_function, sd_prime: real_function) -> real_function:
    """all the input and output functions on the frequency"""
    values = lambda freq: sd_prime(freq) / sd(freq)
    return lambda freq: np.mean(values(freq)) ** 2


def p(z: complex, c: float) -> complex:
    """
    Stieltjes transform of the distribution D
    """
    return -c * z_t_t_tilde(z, c) ** 3 / (1 - c * z_t_t_tilde(z, c) ** 2)


def compute_thetas(
    y: f64_2d,
    B: int,
    f: real_function,
    n_max_freqs: int,
    sd: Optional[real_function] = None,
    L: Optional[int] = None,
    f_against_mp: Optional[None] =None, 
    f_against_D: Optional[None]=None,
) -> f64_1d:
    """
    Compute theta for each frequency
    """
    assert (L is not None) or (sd is not None)

    # start by computing the LSS of f on the spectral coherencies
    # estimate the spectral coherence from the sample
    C_hats, freqs = coherence(y, B, n_max_freqs=n_max_freqs)

    # n_freqs may be less than n_samples if n_max_freqs is not None and n_max_freqs < n_samples
    n_freqs = len(freqs)

    # for each C_hat, compute the Linear Spectral Statistics associated with f
    eigenvalues = [np.linalg.eigvalsh(C_hat) for C_hat in C_hats]
    LSSs = np.array([np.mean(f(eigenvalues[n])) for n in range(n_freqs)])

    # computhe corrective terms
    N, M = y.shape
    c = M / (B + 1)

    # MP acting on f
    if f_against_mp is None: 
        f_against_mp = action_D_on_f(f, lambda z: t(z, c), support_MP(c))

    # correction D acting on f
    if f_against_D is None: 
        f_against_D = action_D_on_f(f, lambda z: p(z, c), support_MP(c))

    # last the constants v_n and r_n
    v_n = _v_n(B, N)

    # compute the second corrective terms: r_n. Estimate the spectral densities of the
    # time series if not provided.
    if sd is None:
        sd = lag_window(y, L)
    sd_prime = derivative(sd)
    r_n_function = _r_n(sd, sd_prime)
    r_n = np.array([r_n_function(freq) for freq in freqs])

    # print(f_against_mp, f_against_D)
    # print(np.mean(np.abs(LSSs)))
    # print(np.mean(np.abs(LSSs - f_against_mp)))
    # print(np.mean(np.abs(LSSs - f_against_mp - f_against_D * (r_n * v_n - 1 / (c * B)))))

    return LSSs - f_against_mp - f_against_D * (r_n * v_n - 1 / (c * B))
