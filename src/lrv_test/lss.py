import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from spectral_coherence import half_coherences

from lrv_test.types import f64_1d, f64_2d, real_function


def _compute_LSS(hC_hat: f64_2d, f: real_function) -> float:
    hC_hat = np.array(hC_hat)

    # Always compute the svd on the most favorable matrix
    M, B = hC_hat.shape
    if M > B:
        hC_hat = hC_hat.T

    sv = jnp.linalg.svd(hC_hat, full_matrices=False, compute_uv=False)
    λ = sv**2

    return np.mean(f(λ))


def compute_LSSs(
    y: f64_2d, B: int, f: real_function, freqs: f64_1d = None
) -> tuple[f64_1d, f64_1d]:
    hC_hats, freqs = half_coherences(mx.array(y), B, freqs=freqs)
    LSSs = np.array([_compute_LSS(hC_hat, f) for hC_hat in hC_hats])
    return LSSs, freqs
