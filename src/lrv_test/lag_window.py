from typing import Callable

import numpy as np


def _compute_autocors(x: np.array, L: int) -> np.array:
    """
    Compute the autocorrelation of a time series x up to lag L

    Parameters
    ----------
    x : mx.array
        Time series of shape (n_samples, n_features)
    L : int
        Maximum lag

    Returns
    -------
    autocors : mx.array
        Autocorrelation of shape (2L+1, n_features)
    """
    # compute the autocorrelation of the time series (positive, 0 and negative lags)
    # shape of x is L x n_features
    r_l_positive_hats = np.stack(
        [np.mean(x[l:, :] * np.conj(x[:-l, :]), axis=0) for l in range(1, L + 1)]
    )
    r_l_negative_hats = np.conj(r_l_positive_hats[::-1, :])
    r_l_0_hats = np.mean(np.abs(x) ** 2, axis=0)

    # shape will be (2L+1) x n_features
    r_l_hats = np.concatenate(
        [r_l_negative_hats, r_l_0_hats[np.newaxis, :], r_l_positive_hats]
    )
    return r_l_hats


def lag_window(X: np.array, L: int) -> Callable[[float], np.array]:
    """
    Compute the lag window estimator of the spectral density of a time series X.
    """
    n_samples, n_features = X.shape

    # compute the autocorrelation of the time series
    r_l_hats = _compute_autocors(X, L)

    # compute the exponential term
    L_range = np.arange(-L, L + 1)
    exp_sequence = lambda nu: np.exp(-2j * np.pi * L_range * nu)
    exp_term = lambda nu: np.tile(exp_sequence(nu)[:, np.newaxis], (1, n_features))

    return lambda nu: np.real(np.sum(r_l_hats * exp_term(nu), axis=0))
