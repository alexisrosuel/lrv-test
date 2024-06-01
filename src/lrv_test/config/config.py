from dataclasses import dataclass
from typing import Optional

from lrv_test.types import real_function


@dataclass(frozen=True)
class LRVConfig:
    B: int  # size of the smoothing window for the spectral density estimation
    L: int  # number of lags to include in the lag window estimator of the spectral density
    f: real_function  # function to use in the LSS
    n_max_freqs: Optional[int] = (
        None  # number of frequencies to include in the spectral coherence estimator
    )

    # def __post_init__(self):
    #     """Check that the test function is supported. We have computed
    #     - f against MP distribution
    #     - f against D_N distribution
    #     - variance of lss in the clt
    #     for only few functions"""
    #     if self.f.__code__.co_code not in {X_MINUS_1_SQUARE}:
    #         raise NotImplementedError
