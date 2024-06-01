from math import floor
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

from lrv_test.config.config import LRVConfig
from lrv_test.contour import Contour
from lrv_test.result import LRVResult
from lrv_test.sigma import compute_sigma
from lrv_test.theta import compute_thetas
from lrv_test.types import f64_1d, f64_2d, real_function


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
    N, M = y.shape

    if n_max_freqs is not None:
        assert (n_max_freqs > 0) and (n_max_freqs < N)


    thetas = compute_thetas(y, B, f, n_max_freqs, sd, L, f_against_mp, f_against_D)
    if sigma is None: 
        sigma = compute_sigma(f, c=M / (B + 1))

    # print(sigma)
    
    # import pdb; pdb.set_trace()
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

    return B * thetas , sigma

    return LRVResult(
        N,
        M,
        B, 
        # freqs,
        # LSSs,
        # c_n,
        # f_mp,
        # f_Dn,
        # r_n,
        # v_n,
        # sigma_n,
        _thetas, 
        sigma,
        zeta_1,
        zeta_2,
        t_stats_0, 
        t_stat_1,
        t_stat_2,
        p_values_0,
        p_value_1, 
        p_value_2


    )
