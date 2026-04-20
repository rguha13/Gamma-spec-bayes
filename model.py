"""Likelihood and prior utilities for the Bayesian FSA model."""

import numpy as np


def compute_mu(activity, gamma, theta, z, livetime, isotope_interpolators, background_interpolators, channels):
    """Compute the Poisson mean spectrum for one parameter state."""

    signal = np.zeros_like(channels, dtype=float)
    for j, activity_j in enumerate(activity):
        if z[j] == 1 and activity_j > 0:
            signal += activity_j * isotope_interpolators[j](channels + theta[j])

    background = np.zeros_like(channels, dtype=float)
    for b, gamma_b in enumerate(gamma):
        if gamma_b > 0:
            background += gamma_b * background_interpolators[b](channels)

    return np.clip(livetime * (signal + background), 1e-12, None)


def log_likelihood(counts, activity, gamma, theta, z, livetime, isotope_interpolators, background_interpolators, channels):
    """Poisson log-likelihood up to the count-factorial constant."""

    mu = compute_mu(
        activity,
        gamma,
        theta,
        z,
        livetime,
        isotope_interpolators,
        background_interpolators,
        channels,
    )
    return float(np.sum(counts * np.log(mu) - mu))


def inverse_gamma_log_prior(x, alpha, beta):
    """Inverse-gamma log density up to the normalizing constant."""

    if x <= 0:
        return -np.inf
    return -((alpha + 1.0) * np.log(x)) - beta / x


def half_normal_log_prior(x, scale):
    """Half-normal log density up to the normalizing constant."""

    if x <= 0:
        return -np.inf
    return -0.5 * (x**2) / (scale**2)


def truncated_normal_log_prior(x, sigma, lower=-2.0, upper=2.0):
    """Truncated normal log density up to the normalizing constant."""

    if x < lower or x > upper:
        return -np.inf
    return -0.5 * (x**2) / (sigma**2)

