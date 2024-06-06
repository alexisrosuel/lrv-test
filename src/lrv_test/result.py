from dataclasses import dataclass
from typing import Optional

import numpy as np

from lrv_test.config.config import LRVConfig
from lrv_test.types import f64_1d, real_function


@dataclass(frozen=True)
class LRVResult:
    N: int
    M: int
    B: int 
    # L: int
    freqs: np.ndarray  # frequencies at which the LSS is evaluated
    LSSs: np.ndarray  # values of the LSS at the frequencies
    thetas: np.ndarray 
    f_mp: float
    corrections: float
    # r_n: f64_1d  # values of the r_n_hat term at the frequencies
    # v_n: float  # value of the v_n term
    # sigma_N: float  # value of sigma_N(f) for the given f
    t_stat_0: float
    t_stat_1: float 
    t_stat_2: float

    # @property
    # def B(self) -> int:
    #     return self.lrv_config.B

    # @property
    # def L(self) -> int:
    #     return self.lrv_config.L

    # @property
    # def f(self) -> real_function:
    #     return self.config.f

    # @property
    # def n_max_freqs(self) -> Optional[int]:
    #     return self.config.n_max_freqs

    # @property
    # def correction_term_1(self) -> np.ndarray:
    #     return self.f_Dn * self.r_n * self.v_n

    # @property
    # def correction_term_2(self) -> np.ndarray:
    #     return -self.f_Dn * 1 / self.c_n * 1 / self.B

    # @property
    # def theta_n(self) -> Optional[np.ndarray]:
    #     if self.correction_term_1 is None:
    #         return None
    #     return self.LSSs - self.f_mp - self.correction_term_1 - self.correction_term_2

    # @property
    # def zeta_1_n(self) -> Optional[np.ndarray]:
    #     if self.theta_n is None:
    #         return None
    #     return np.sum(self.B * self.theta_n) / np.sqrt(len(self.theta_n))

    # @property
    # def zeta_2_n(self) -> Optional[np.ndarray]:
    #     if self.theta_n is None:
    #         return None
    #     return np.sum((self.B * self.theta_n) ** 2) / len(self.theta_n)
