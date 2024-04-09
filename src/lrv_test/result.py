from dataclasses import dataclass
from typing import Optional

import numpy as np

from lrv_test.config import LRVConfig
from lrv_test.types import f64_1d, real_function


@dataclass(frozen=True)
class LRVResult:
    N: int
    M: int
    lrv_config: LRVConfig
    freqs: np.ndarray  # frequencies at which the LSS is evaluated
    LSSs: np.ndarray  # values of the LSS at the frequencies
    c_n: float
    f_mp: float
    f_D_n: float
    r_n: f64_1d  # values of the r_n_hat term at the frequencies
    v_n: float  # value of the v_n term
    sigma_N: float  # value of sigma_N(f) for the given f

    @property
    def B(self) -> int:
        return self.lrv_config.B

    @property
    def L(self) -> int:
        return self.lrv_config.L

    @property
    def f(self) -> real_function:
        return self.config.f

    @property
    def n_max_freqs(self) -> Optional[int]:
        return self.config.n_max_freqs

    @property
    def correction_term_1(self) -> Optional[np.ndarray]:
        if self.r_n is None:
            return None
        return self.f_D_n * self.r_n * self.v_n

    @property
    def correction_term_2(self) -> np.ndarray:
        return -self.f_D_n * 1 / self.c_n * 1 / self.B

    @property
    def correction_term_1_hat(self) -> np.ndarray:
        return self.f_D_n * self.r_n_hat * self.v_n

    @property
    def theta_n(self) -> Optional[np.ndarray]:
        if self.correction_term_1 is None:
            return None
        return self.LSSs - self.f_mp - self.correction_term_1 - self.correction_term_2

    @property
    def theta_n_hat(self) -> np.ndarray:
        return (
            self.LSSs - self.f_mp - self.correction_term_1_hat - self.correction_term_2
        )

    @property
    def zeta_1_n(self) -> Optional[np.ndarray]:
        if self.theta_n is None:
            return None
        return np.sum(self.B * self.theta_n) / np.sqrt(len(self.theta_n))

    @property
    def zeta_1_n_hat(self) -> np.ndarray:
        return np.sum(self.B * self.theta_n_hat) / np.sqrt(len(self.theta_n_hat))

    @property
    def zeta_2_n(self) -> Optional[np.ndarray]:
        if self.theta_n is None:
            return None
        return np.sum((self.B * self.theta_n) ** 2) / len(self.theta_n)

    @property
    def zeta_2_n_hat(self) -> np.ndarray:
        return np.sum((self.B * self.theta_n_hat) ** 2 - self.sigma_N**2) / len(
            self.theta_n_hat
        )
