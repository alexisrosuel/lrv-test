from typing import Optional

import numpy as np
from scipy.special import gamma

from lrv_test.contour import Contour
from lrv_test.functions import support_MP, t
from lrv_test.lag_window import lag_window
from lrv_test.lss import compute_LSSs
from lrv_test.result import LRVResult
from lrv_test.sigma import compute_sigma
from lrv_test.types import f64_2d, real_function
from lrv_test.utils import action_D_on_f, contour_integral, derivative, psi


def _v_n(B: int, N: int) -> float:
    b_range = np.arange(-(B - 1) // 2, (B - 1) // 2 + 1)
    return np.mean((b_range / N) ** 2)


def _r_n(sd: real_function, sd_prime: real_function) -> real_function:
    """
    all the input and output are functions on the frequency
    """
    values = lambda freq: sd_prime(freq) / sd(freq)
    return lambda freq: np.mean(values(freq)) ** 2


def LRV(
    y: f64_2d,
    B: int,
    f: real_function,
    freqs: Optional[np.ndarray] = None,
    L: Optional[int] = None,
    sd: Optional[real_function] = None,
    f_against_mp: Optional[float] = None,
    f_against_D: Optional[float] = None,
    sigma: Optional[float] = None,
    tolerance: float = 1e-6,
) -> LRVResult:
    """
    Compute the LRV statistics on the time series y. The LRVResult object also
    contains additional details about the statistics computation.
    """
    # either L or the true spectral density will be used to compute the r_n(nu).
    # If none is provided, the correction term proportional to r_n(nu) will be skipped.
    if L is None and sd is None:
        skip_correction = True
    else:
        skip_correction = False

    N, M = y.shape

    # Compute the LSSs associated with each coherency matrix
    LSSs, freqs = compute_LSSs(y, B, f, freqs)

    # compute corrective terms (MP acting and f, and D acting on f)
    c = M / B
    if f_against_mp is None:
        f_against_mp = action_D_on_f(f, lambda z: t(z, c), support_MP(c), tolerance)
    if f_against_D is None:
        support = (-np.sqrt(c), np.sqrt(c))
        radius = (support[1] - support[0]) / 2
        center = (support[0] + support[1]) / 2
        contour = Contour.from_circle_parameters(center, radius)
        f_against_D = contour_integral(
            lambda w: -c / (2 * np.pi * 1j) * f(psi(w, c)) / (w**3), contour
        )
        assert np.imag(f_against_D) < tolerance
        f_against_D = np.real(f_against_D)

    # Estimate the spectral densities of the time series if not provided.
    if skip_correction:
        r_n = 0
    else:
        if sd is None:
            sd = lag_window(y, L)
        sd_prime = derivative(sd)
        r_n_function = _r_n(sd, sd_prime)
        r_n = np.array([r_n_function(freq) for freq in freqs])

    v_n = _v_n(B, N)

    # compute thetas (are asymptoticaly gaussian under H0)
    corrections = f_against_D * (r_n * v_n - 1 / (c * B))
    thetas = LSSs - f_against_mp - corrections

    # compute the limit variance of each thetas if not provided
    if sigma is None:
        sigma = compute_sigma(f, c, tolerance)

    # compute the LRV test statistics
    n_freqs = len(thetas)
    # t_stats_0, 1 and 2 are  N(0,1) asymptotically
    t_stats_0 = M * thetas / sigma
    t_stat_1 = (1 / np.sqrt(n_freqs)) * np.sum(t_stats_0)
    t_stat_2 = (
        (1 / np.sqrt(n_freqs))
        * (np.sum((M * thetas) ** 2 - sigma**2))
        / (np.sqrt(2) * sigma**2)
    )
    t_stat_3 = np.sum(t_stats_0**2)  # is χ²(n_freqs) asymptotically
    a_n = 1 / 2
    b_n = 2 * (np.log(n_freqs) - 0.5 * np.log(np.log(n_freqs)) - np.log(gamma(1 / 2)))
    t_stat_4 = a_n * (np.max(t_stats_0**2) - b_n)  # is conjectured Gumbel distribution

    return LRVResult(
        N,
        M,
        B,
        freqs,
        LSSs,
        v_n,
        thetas,
        f_against_mp,
        corrections,
        t_stats_0,
        t_stat_1,
        t_stat_2,
        t_stat_3,
        t_stat_4,
    )
