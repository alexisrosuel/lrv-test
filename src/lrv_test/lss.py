
import numpy as np
from spectral_coherence import coherence

from lrv_test.types import f64_1d, f64_2d, real_function


def _compute_LSS(
    C_hat: f64_2d,
    f: real_function
) -> float:
    eigs = np.linalg.eigvalsh(C_hat)
    return np.mean(f(eigs))


def compute_LSSs(y: f64_2d, B: int, f: real_function, n_max_freqs: int) -> tuple[f64_1d, f64_1d]:
    # estimate the spectral coherence from the sample
    C_hats, freqs = coherence(y, B, n_max_freqs=n_max_freqs)

    # for each C_hat, compute the Linear Spectral Statistics associated with f
    LSSs = np.array([_compute_LSS(C_hat, f) for C_hat in C_hats])

    return LSSs, freqs
