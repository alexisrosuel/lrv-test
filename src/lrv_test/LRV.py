from typing import Optional

import numpy as np
from scipy.stats import norm
from spectral_coherence import coherence, lag_window

from lrv_test.contour import Contour
from lrv_test.functions import support_MP, t, z_t_t_tilde
from lrv_test.lag_window import lag_window
from lrv_test.lss import compute_LSSs
from lrv_test.result import LRVResult
from lrv_test.sigma import compute_sigma
from lrv_test.types import f64_1d, f64_2d, real_function
from lrv_test.utils import action_D_on_f, contour_integral, derivative, psi


def _v_n(B: int, N: int) -> float:
    b_range = np.arange(-(B - 1) // 2, (B - 1) // 2 + 1)
    return np.mean((b_range / N) ** 2)


def _r_n(sd: real_function, sd_prime: real_function) -> real_function:
    """all the input and output functions on the frequency"""
    values = lambda freq: sd_prime(freq) / sd(freq)
    return lambda freq: np.mean(values(freq)) ** 2


# def p(z: complex, c: float) -> complex:
#     """
#     Stieltjes transform of the distribution D
#     """
#     _z_t_t_tilde = z_t_t_tilde(z, c)
#     return -c * _z_t_t_tilde ** 3 / (1 - c * _z_t_t_tilde ** 2)



def LRV(
    y: f64_2d,
    B: int,
    f: real_function,
    n_max_freqs: Optional[int] = None,
    L: Optional[int] = None,
    sd: Optional[real_function] = None,
    f_against_mp: Optional[float] = None ,
    f_against_D: Optional[float] = None ,
    sigma: Optional[float] = None, 
) -> LRVResult:
    """
    Compute the LRV statistics on the time series y. The LRVResult object also
    contains additional details about the statistics computation.
    """
    # either L or the true spectral density will be used to compute r_n
    assert (L is not None) or (sd is not None)

    N, M = y.shape
    if n_max_freqs is not None:
        assert (n_max_freqs > 0) and (n_max_freqs < N)
    
    LSSs, freqs = compute_LSSs(y, B, f, n_max_freqs)

    # compute corrective terms
    c = M / (B + 1)

    # MP acting on f
    if f_against_mp is None: 
        f_against_mp = action_D_on_f(f, lambda z: t(z, c), support_MP(c))

    # correction D acting on f
    if f_against_D is None: 
        support = (-np.sqrt(c)-0.1, np.sqrt(c)+0.1)
        radius = (support[1] - support[0]) / 2  
        center = (support[0] + support[1]) / 2
        contour = Contour.from_circle_parameters(center, radius)
        f_against_D = contour_integral(lambda w: -c / (2*np.pi*1j) * f(psi(w,c)) / (w**3), contour)
        f_against_D = np.real(f_against_D)

    # Estimate the spectral densities of the time series if not provided.
    if sd is None:
        sd = lag_window(y, L)
    sd_prime = derivative(sd)

    # constants r_n and v_n
    r_n_function = _r_n(sd, sd_prime)
    r_n = np.array([r_n_function(freq) for freq in freqs])
    v_n = _v_n(B, N)

    # compute thetas
    correction = f_against_D * (r_n * v_n - 1 / (c * B))
    thetas = LSSs - f_against_mp - correction

    # compute sigma if not provided
    if sigma is None: 
        sigma = compute_sigma(f, c=M / (B + 1))
    
    zeta_1 = 1 / np.sqrt(B) * np.sum(B * thetas)
    zeta_2 = 1 / np.sqrt(B) * np.sum((B * thetas) ** 2 - sigma**2)

    # compute the variables that are asymptotically N(0,1)
    t_stats_0 = B * thetas / sigma
    t_stat_1 = zeta_1 / sigma
    t_stat_2 = zeta_2 / (np.sqrt(2) * sigma**2)

    # compute the corresponding p_values
    p_values_0 = 1 - norm.cdf(t_stats_0)
    p_value_1 = 1 - norm.cdf(t_stat_1)
    p_value_2 = 1 - norm.cdf(t_stat_2)

    return LRVResult(
        N,
        M,
        B,
        # L, 
        freqs,
        LSSs,
        # c_n,
        # f_mp,
        # f_Dn,
        # r_n,
        # v_n,
        # sigma_n,
        thetas, 
        f_against_mp, 
        correction,
        # sigma,
        # zeta_1,
        # zeta_2,
        t_stats_0, 
        t_stat_1,
        t_stat_2,
        # p_values_0,
        # p_value_1, 
        # p_value_2


    )
