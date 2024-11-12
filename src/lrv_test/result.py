from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from scipy.stats import chi2, gumbel_r, norm

from lrv_test.types import f64_1d


def is_positive(
    test_stat: Union[float, np.ndarray],
    distribution: str,
    alternative: Literal["left", "right", "double"],
    level: float,
    df: int = None,
) -> Union[bool, np.ndarray]:
    """
    Check if test statistic is in the rejection region for given level and alternative.

    Args:
        test_stat: Test statistic (scalar or array)
        distribution: One of 'normal', 'chi2', 'gumbel'
        alternative: 'left', 'right', or 'double'
        level: Significance level (e.g., 0.05)
        df: Degrees of freedom (only for chi2 distribution)
    """
    if distribution == "normal":
        if alternative == "left":
            critical_value = norm.ppf(level)
            return test_stat < critical_value
        elif alternative == "right":
            critical_value = norm.ppf(1 - level)
            return test_stat > critical_value
        else:  # double
            critical_value = norm.ppf(1 - level / 2)
            return np.abs(test_stat) > critical_value

    elif distribution == "chi2":
        if alternative == "left":
            critical_value = chi2.ppf(level, df)
            return test_stat < critical_value
        elif alternative == "right":
            critical_value = chi2.ppf(1 - level, df)
            return test_stat > critical_value
        else:  # double
            lower = chi2.ppf(level / 2, df)
            upper = chi2.ppf(1 - level / 2, df)
            return (test_stat < lower) | (test_stat > upper)

    elif distribution == "gumbel":
        if alternative == "left":
            critical_value = gumbel_r.ppf(level)
            return test_stat < critical_value
        elif alternative == "right":
            critical_value = gumbel_r.ppf(1 - level)
            return test_stat > critical_value
        else:  # double
            lower = gumbel_r.ppf(level / 2)
            upper = gumbel_r.ppf(1 - level / 2)
            return (test_stat < lower) | (test_stat > upper)

    raise ValueError(f"Unknown distribution: {distribution}")


@dataclass(frozen=True)
class LRVResult:
    N: int
    M: int
    B: int
    freqs: np.ndarray  # frequencies at which the LSS is evaluated
    LSSs: np.ndarray  # values of the LSS at the frequencies
    v_n: float
    thetas: np.ndarray
    f_mp: float
    corrections: float
    t_stats_0: f64_1d
    t_stat_1: float
    t_stat_2: float
    t_stat_3: float
    t_stat_4: float

    def is_positive_0(
        self, level: float, alternative: Literal["left", "right", "double"] = "double"
    ) -> np.ndarray:
        return is_positive(self.t_stats_0, "normal", alternative, level)

    def is_positive_1(
        self, level: float, alternative: Literal["left", "right", "double"] = "double"
    ) -> bool:
        return is_positive(self.t_stat_1, "normal", alternative, level)

    def is_positive_2(
        self, level: float, alternative: Literal["left", "right", "double"] = "double"
    ) -> bool:
        return is_positive(self.t_stat_2, "normal", alternative, level)

    def is_positive_3(
        self, level: float, alternative: Literal["left", "right", "double"] = "right"
    ) -> bool:
        return is_positive(
            self.t_stat_3, "chi2", alternative, level, df=len(self.freqs)
        )

    def is_positive_4(
        self, level: float, alternative: Literal["left", "right", "double"] = "right"
    ) -> bool:
        return is_positive(self.t_stat_4, "gumbel", alternative, level)
