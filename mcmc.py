"""Metropolis-within-Gibbs sampler for the Bayesian FSA model."""

from dataclasses import dataclass

import numpy as np

from .config import MCMCConfig, PriorConfig, ProposalConfig
from .model import (
    half_normal_log_prior,
    inverse_gamma_log_prior,
    log_likelihood,
    truncated_normal_log_prior,
)


@dataclass
class ChainResult:
    """Post-burn-in samples and acceptance rates for one chain."""

    activity: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    z: np.ndarray
    loglik: np.ndarray
    acc_activity: np.ndarray
    acc_gamma: np.ndarray
    acc_theta: np.ndarray
    acc_z: np.ndarray


@dataclass
class MultiChainResult:
    """Post-burn-in samples from multiple chains."""

    activity: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    z: np.ndarray
    loglik: np.ndarray


def sample_inverse_gamma(rng, alpha, beta):
    """Sample from IG(alpha, beta), where beta is the scale parameter."""

    return 1.0 / rng.gamma(shape=alpha, scale=1.0 / beta)


def sample_truncated_normal(rng, size, sigma, lower=-2.0, upper=2.0):
    """Rejection-sample a normal distribution truncated to [lower, upper]."""

    values = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        draws = rng.normal(0.0, sigma, size=size - filled)
        draws = draws[(draws >= lower) & (draws <= upper)]
        n_new = draws.size
        if n_new:
            values[filled : filled + n_new] = draws
            filled += n_new
    return values


def initialize_state(k_isotopes, n_background, prior: PriorConfig, rng):
    """Draw an overdispersed initial state from the model priors."""

    pi = prior.inclusion_probability
    z = rng.binomial(1, pi, size=k_isotopes).astype(int)
    activity = np.zeros(k_isotopes, dtype=float)
    for j in range(k_isotopes):
        if z[j] == 1:
            activity[j] = sample_inverse_gamma(rng, prior.alpha_ig, prior.beta_ig)

    gamma = np.clip(
        np.abs(rng.normal(0.0, prior.gamma_scale, size=n_background)),
        1e-6,
        None,
    )
    theta = sample_truncated_normal(rng, k_isotopes, prior.sigma_theta_channels)
    return activity, gamma, theta, z


def update_activity_j(j, state, counts, livetime, templates, prior, proposal, rng):
    """Random-walk Metropolis update for one active isotope activity."""

    activity, gamma, theta, z = state
    if z[j] == 0 or activity[j] <= 0:
        return False

    current = activity[j]
    ll_current = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_current = inverse_gamma_log_prior(current, prior.alpha_ig, prior.beta_ig)

    proposed = current * np.exp(proposal.step_log_activity * rng.normal())
    activity[j] = proposed
    ll_proposed = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_proposed = inverse_gamma_log_prior(proposed, prior.alpha_ig, prior.beta_ig)

    # The final term is the Jacobian correction for proposing on log(A_j).
    log_accept = (ll_proposed + lp_proposed) - (ll_current + lp_current) + np.log(proposed / current)

    if np.log(rng.uniform()) < log_accept:
        return True

    activity[j] = current
    return False


def update_gamma_b(b, state, counts, livetime, templates, prior, proposal, rng):
    """Random-walk Metropolis update for one background coefficient."""

    activity, gamma, theta, z = state
    current = gamma[b]
    if current <= 0:
        return False

    ll_current = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_current = half_normal_log_prior(current, prior.gamma_scale)

    proposed = current + rng.normal(0.0, proposal.step_gamma)
    if proposed <= 0:
        return False

    gamma[b] = proposed
    ll_proposed = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_proposed = half_normal_log_prior(proposed, prior.gamma_scale)
    log_accept = (ll_proposed + lp_proposed) - (ll_current + lp_current)

    if np.log(rng.uniform()) < log_accept:
        return True

    gamma[b] = current
    return False


def update_theta_j(j, state, counts, livetime, templates, prior, proposal, rng):
    """Random-walk Metropolis update for one isotope shift parameter."""

    activity, gamma, theta, z = state
    current = theta[j]
    ll_current = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_current = truncated_normal_log_prior(current, prior.sigma_theta_channels)

    proposed = current + rng.normal(0.0, proposal.step_theta)
    lp_proposed = truncated_normal_log_prior(proposed, prior.sigma_theta_channels)
    if not np.isfinite(lp_proposed):
        return False

    theta[j] = proposed
    ll_proposed = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    log_accept = (ll_proposed + lp_proposed) - (ll_current + lp_current)

    if np.log(rng.uniform()) < log_accept:
        return True

    theta[j] = current
    return False


def flip_z_j(j, state, counts, livetime, templates, prior, rng):
    """Metropolis update for one isotope inclusion indicator.

    When switching from off to on, A_j is proposed from the inverse-gamma slab
    prior. The slab prior density therefore cancels the proposal density in the
    Metropolis-Hastings ratio, leaving only the likelihood and Bernoulli prior
    terms.
    """

    activity, gamma, theta, z = state
    current_z = z[j]
    current_a = activity[j]
    ll_current = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)

    pi = prior.inclusion_probability
    lp_z_current = current_z * np.log(pi) + (1 - current_z) * np.log(1 - pi)

    if current_z == 0:
        proposed_z = 1
        proposed_a = sample_inverse_gamma(rng, prior.alpha_ig, prior.beta_ig)
    else:
        proposed_z = 0
        proposed_a = 0.0

    z[j] = proposed_z
    activity[j] = proposed_a

    ll_proposed = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)
    lp_z_proposed = proposed_z * np.log(pi) + (1 - proposed_z) * np.log(1 - pi)
    log_accept = (ll_proposed + lp_z_proposed) - (ll_current + lp_z_current)

    if np.log(rng.uniform()) < log_accept:
        return True

    z[j] = current_z
    activity[j] = current_a
    return False


def run_single_chain(counts, livetime, template_library, prior, proposal, mcmc, seed, verbose=True):
    """Run one Metropolis-within-Gibbs chain and return post-burn-in samples."""

    rng = np.random.default_rng(seed)
    k_isotopes = len(template_library.isotope_interpolators)
    n_background = len(template_library.background_interpolators)
    templates = (
        template_library.isotope_interpolators,
        template_library.background_interpolators,
        template_library.channels,
    )

    activity, gamma, theta, z = initialize_state(k_isotopes, n_background, prior, rng)
    n_iter = mcmc.n_iter
    burnin = mcmc.burnin
    n_keep = n_iter - burnin

    samples_activity = np.zeros((n_iter, k_isotopes), dtype=float)
    samples_gamma = np.zeros((n_iter, n_background), dtype=float)
    samples_theta = np.zeros((n_iter, k_isotopes), dtype=float)
    samples_z = np.zeros((n_iter, k_isotopes), dtype=int)
    samples_loglik = np.zeros(n_iter, dtype=float)

    acc_activity = np.zeros(k_isotopes, dtype=float)
    acc_gamma = np.zeros(n_background, dtype=float)
    acc_theta = np.zeros(k_isotopes, dtype=float)
    acc_z = np.zeros(k_isotopes, dtype=float)

    for iteration in range(n_iter):
        state = (activity, gamma, theta, z)

        for j in range(k_isotopes):
            accepted = flip_z_j(j, state, counts, livetime, templates, prior, rng)
            if iteration >= burnin:
                acc_z[j] += int(accepted)

        for j in range(k_isotopes):
            accepted = update_activity_j(j, state, counts, livetime, templates, prior, proposal, rng)
            if iteration >= burnin:
                acc_activity[j] += int(accepted)

        for b in range(n_background):
            accepted = update_gamma_b(b, state, counts, livetime, templates, prior, proposal, rng)
            if iteration >= burnin:
                acc_gamma[b] += int(accepted)

        for j in range(k_isotopes):
            accepted = update_theta_j(j, state, counts, livetime, templates, prior, proposal, rng)
            if iteration >= burnin:
                acc_theta[j] += int(accepted)

        samples_activity[iteration] = activity
        samples_gamma[iteration] = gamma
        samples_theta[iteration] = theta
        samples_z[iteration] = z
        samples_loglik[iteration] = log_likelihood(counts, activity, gamma, theta, z, livetime, *templates)

        if verbose and ((iteration + 1) % 2000 == 0 or iteration == n_iter - 1):
            print(f"  Seed {seed}: iteration {iteration + 1}/{n_iter}")

    return ChainResult(
        activity=samples_activity[burnin:],
        gamma=samples_gamma[burnin:],
        theta=samples_theta[burnin:],
        z=samples_z[burnin:],
        loglik=samples_loglik[burnin:],
        acc_activity=acc_activity / n_keep,
        acc_gamma=acc_gamma / n_keep,
        acc_theta=acc_theta / n_keep,
        acc_z=acc_z / n_keep,
    )


def run_multiple_chains(counts, livetime, template_library, prior: PriorConfig, proposal: ProposalConfig, mcmc: MCMCConfig, verbose=True):
    """Run multiple independent chains and stack post-burn-in samples."""

    mcmc.validate()
    k_isotopes = len(template_library.isotope_interpolators)
    n_background = len(template_library.background_interpolators)
    n_keep = mcmc.n_iter - mcmc.burnin

    chains_activity = np.zeros((mcmc.n_chains, n_keep, k_isotopes), dtype=float)
    chains_gamma = np.zeros((mcmc.n_chains, n_keep, n_background), dtype=float)
    chains_theta = np.zeros((mcmc.n_chains, n_keep, k_isotopes), dtype=float)
    chains_z = np.zeros((mcmc.n_chains, n_keep, k_isotopes), dtype=int)
    chains_loglik = np.zeros((mcmc.n_chains, n_keep), dtype=float)

    for chain in range(mcmc.n_chains):
        seed = mcmc.base_seed + 1000 * chain
        if verbose:
            print(f"\nStarting chain {chain + 1}/{mcmc.n_chains} with seed {seed}")
        result = run_single_chain(
            counts=counts,
            livetime=livetime,
            template_library=template_library,
            prior=prior,
            proposal=proposal,
            mcmc=mcmc,
            seed=seed,
            verbose=verbose,
        )
        chains_activity[chain] = result.activity
        chains_gamma[chain] = result.gamma
        chains_theta[chain] = result.theta
        chains_z[chain] = result.z
        chains_loglik[chain] = result.loglik

    return MultiChainResult(
        activity=chains_activity,
        gamma=chains_gamma,
        theta=chains_theta,
        z=chains_z,
        loglik=chains_loglik,
    )

