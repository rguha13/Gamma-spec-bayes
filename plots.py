"""Plotting helpers for Bayesian FSA outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import compute_mu
from .summaries import pooled_samples


def _savefig(fig, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_reconstruction(counts, livetime, result, template_library, energy, output_dir: Path) -> None:
    """Plot observed count rate and posterior-mean reconstruction."""

    pooled = pooled_samples(result)
    activity_mean = pooled["activity"].mean(axis=0)
    gamma_mean = pooled["gamma"].mean(axis=0)
    theta_mean = pooled["theta"].mean(axis=0)
    z_mean = (pooled["z"].mean(axis=0) >= 0.5).astype(int)

    mu_mean = compute_mu(
        activity_mean,
        gamma_mean,
        theta_mean,
        z_mean,
        livetime,
        template_library.isotope_interpolators,
        template_library.background_interpolators,
        template_library.channels,
    )
    rate_obs = counts / livetime
    rate_fit = mu_mean / livetime

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(energy, rate_obs, where="mid", label="Observed", alpha=0.75)
    ax.step(energy, rate_fit, where="mid", label="Posterior mean", alpha=0.75)
    mask = energy <= 1500
    ax.set_xlim(energy[mask][0], 1500)
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Count rate (s$^{-1}$)")
    ax.legend(frameon=False)
    _savefig(fig, output_dir, "spectral_reconstruction.png")


def plot_activity_traces(result, isotope_names, selected_idx, output_dir: Path) -> None:
    """Plot per-chain activity traces for selected isotopes."""

    if len(selected_idx) == 0:
        return

    n_chains, n_draws, _ = result.activity.shape
    draw_idx = np.arange(n_draws)
    fig, axes = plt.subplots(len(selected_idx), 1, figsize=(7.16, 2.0 * len(selected_idx)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, j in zip(axes, selected_idx):
        for chain in range(n_chains):
            ax.plot(draw_idx, result.activity[chain, :, j] / 1e3, linewidth=0.7, alpha=0.8)
        ax.set_ylabel(f"{isotope_names[j]}\n(kBq)")

    axes[-1].set_xlabel("Post-burn-in draw")
    _savefig(fig, output_dir, "trace_activities.png")


def plot_activity_posteriors(result, isotope_names, selected_idx, output_dir: Path) -> None:
    """Plot posterior marginal histograms for selected isotope activities."""

    if len(selected_idx) == 0:
        return

    activity_kbq = pooled_samples(result)["activity"] / 1e3
    n_cols = 2 if len(selected_idx) <= 4 else 3
    n_rows = int(np.ceil(len(selected_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 1.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_index, j in enumerate(selected_idx):
        ax = axes[ax_index]
        values = activity_kbq[:, j]
        mean = values.mean()
        lo, hi = np.percentile(values, [2.5, 97.5])
        ax.hist(values, bins=40, density=True, color="steelblue", edgecolor="black", linewidth=0.4)
        ax.axvline(mean, linewidth=1.0, color="black", label="Mean")
        ax.axvline(lo, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(hi, linestyle="--", linewidth=0.8, color="gray", label="95% CI")
        ax.set_title(isotope_names[j])
        ax.set_xlabel("Activity (kBq)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, frameon=False)

    for ax in axes[len(selected_idx):]:
        ax.axis("off")
    _savefig(fig, output_dir, "posterior_activities.png")


def plot_shift_posteriors(result, isotope_names, selected_idx, b_calibration, output_dir: Path) -> None:
    """Plot posterior marginal histograms for selected isotope shifts in keV."""

    if len(selected_idx) == 0:
        return

    theta_kev = pooled_samples(result)["theta"] * b_calibration
    n_cols = 2 if len(selected_idx) <= 4 else 3
    n_rows = int(np.ceil(len(selected_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 1.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_index, j in enumerate(selected_idx):
        ax = axes[ax_index]
        values = theta_kev[:, j]
        mean = values.mean()
        lo, hi = np.percentile(values, [2.5, 97.5])
        ax.hist(values, bins=40, density=True, color="darkorange", edgecolor="black", linewidth=0.4)
        ax.axvline(mean, linewidth=1.0, color="black")
        ax.axvline(lo, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(hi, linestyle="--", linewidth=0.8, color="gray")
        ax.set_title(isotope_names[j])
        ax.set_xlabel("Shift (keV)")
        ax.set_ylabel("Density")

    for ax in axes[len(selected_idx):]:
        ax.axis("off")
    _savefig(fig, output_dir, "posterior_shifts.png")


def plot_background_posteriors(result, output_dir: Path) -> None:
    """Plot posterior marginal histograms for background coefficients."""

    gamma = pooled_samples(result)["gamma"]
    n_background = gamma.shape[1]
    n_cols = min(n_background, 2)
    n_rows = int(np.ceil(n_background / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.16, 1.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for b in range(n_background):
        ax = axes[b]
        values = gamma[:, b]
        mean = values.mean()
        lo, hi = np.percentile(values, [2.5, 97.5])
        ax.hist(values, bins=40, density=True, color="seagreen", edgecolor="black", linewidth=0.4)
        ax.axvline(mean, linewidth=1.0, color="black")
        ax.axvline(lo, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(hi, linestyle="--", linewidth=0.8, color="gray")
        ax.set_title(f"gamma_{b}")
        ax.set_xlabel("Background coefficient")
        ax.set_ylabel("Density")

    for ax in axes[n_background:]:
        ax.axis("off")
    _savefig(fig, output_dir, "posterior_backgrounds.png")


def plot_isotope_templates(template_library, isotope_names, energy, output_dir: Path) -> None:
    """Plot the averaged isotope template library on a log scale."""

    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    templates = np.clip(template_library.averaged_isotope_templates, 1e-12, None)
    for j, isotope in enumerate(isotope_names):
        ax.step(energy, templates[:, j], where="mid", linewidth=0.9, alpha=0.85, label=isotope)

    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Template response (cps/Bq)")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    _savefig(fig, output_dir, "isotope_templates.png")

