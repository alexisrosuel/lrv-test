from typing import Callable

import numpy as np
from lrv_test.types import f64_2d


def _compute_autocors(x: f64_2d, L: int) -> f64_2d:
    """
    Compute the autocorrelation of a time series x up to lag L

    Parameters
    ----------
    x : np.ndarray
        Time series of shape (n_samples, n_features)
    L : int
        Maximum lag

    Returns
    -------
    autocors : np.ndarray
        Autocorrelation of shape (2L+1, n_features)
    """
    n_samples, n_features = x.shape

    # compute the autocorrelation of the time series
    r_l_positive_hats = np.array(
        [np.mean(x[l:, :] * np.conj(x[:-l, :]), axis=0) for l in range(1, L + 1)]
    )  # shape is L x n_features
    r_l_negative_hats = np.conj(r_l_positive_hats[::-1, :])
    r_l_0_hats = np.mean(np.abs(x) ** 2, axis=0)
    r_l_hats = np.concatenate(
        [r_l_negative_hats, [r_l_0_hats], r_l_positive_hats]
    )  # shape is (2L+1) x n_features

    return r_l_hats


def lag_window(X: f64_2d, L: int) -> Callable[[float], float]:
    """
    Compute the lag window estimator of the spectral density of a time series X.
    """
    n_samples, n_features = X.shape

    # compute the autocorrelation of the time series
    r_l_hats = _compute_autocors(X, L)  # shape is (2L+1) x n_features

    # compute the exponential term
    L_range = np.arange(-L, L + 1)
    exp_sequence = lambda nu: np.exp(-2 * 1j * np.pi * L_range * nu)  # shape is 2L+1
    exp_term = lambda nu: np.tile(
        exp_sequence(nu), (n_features, 1)
    ).T  # shape is (2L+1) x n_features

    return lambda nu: np.real(np.sum(r_l_hats * exp_term(nu), axis=0))
