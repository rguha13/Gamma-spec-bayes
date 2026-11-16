"""Recreate manuscript figures/tables from gamma_spec_bayes.py outputs.

Single-spectrum paper figures (no MCMC is run):
    python plotting.py --results-dir results_updated/<timestamp>/<series>_run1

Only aggregate bias, E_n and selection for a batch of saved analyses:
    python plotting.py --aggregate-root results_updated/<timestamp>

Both commands can be combined. --data-dir defaults to this script's directory;
only Figures 1, 2 and 4 require the input tables and gamma_spec_bayes.py.
Use --skip-context to generate the remaining figures/tables from saved files alone.
--extra-plots adds the diagnostic figures from the old plotting.py that were not
in the manuscript. Outputs are PDF + 300-dpi PNG and full-precision CSV tables.

Tables II--V use 99% equal-tailed intervals, matching the manuscript. Density
plots and E_n use 95% intervals. All statistics use all retained MCMC draws.
"""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, invgamma


# Preserve the original figure layouts and font defaults, with smaller text
# in Figures 2, 5 and 6 and smaller legends in Figure 4.
SINGLE, DOUBLE = 3.5, 7.16
FONT_SIZE, LEGEND_SIZE = 12, 10.5
plt.rcParams.update({
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FONT_SIZE, "figure.labelsize": FONT_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    "axes.linewidth": .6, "lines.linewidth": .8, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
SMALL_FONTS = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
    "figure.titlesize": 8, "figure.labelsize": 8,
}
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")
PEAK_WINDOWS = {
    "Na22": (1245, 1300), "Mn54": (810, 860), "Co57": (100, 150),
    "Co60": (1145, 1360), "Zn65": (1085, 1145), "Y88": (875, 925),
    "Cd109": (65, 115), "Cs134": (775, 825), "Cs137": (635, 685),
    "Pb210": (27, 56), "Am241": (53, 85),
}


def template_names(labels):
    counts, names = {}, {}
    for label in labels:
        isotope = label.split("_", 1)[0]
        counts[isotope] = counts.get(isotope, 0) + 1
        names[label] = f"{isotope}-{ROMAN[counts[isotope] - 1]}"
    return names


def panel_label(ax, label):
    ax.text(.96, .94, label, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=.2", facecolor="white", edgecolor="gray", alpha=.9))


def panels(count, height=2.1, columns=2):
    columns = min(columns, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(DOUBLE if columns > 1 else SINGLE, height * rows))
    for ax in axes.ravel()[count:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:count]


def save_figure(fig, output, name, caption, manifest, rect=None):
    fig.tight_layout(rect=rect, pad=.7)
    # Avoid bbox_inches='tight': the physical width and font sizes stay exact.
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    manifest.append(dict(file=f"{name}.pdf", width_inches=float(fig.get_figwidth()),
                         font_size_pt=plt.rcParams["font.size"],
                         legend_font_size_pt=plt.rcParams["legend.fontsize"], caption=caption))
    plt.close(fig)
    print(f"  {name}", flush=True)


# Load the retained posterior and its fitted grid; never discard burn-in again.
class Posterior:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "metadata.json").read_text())
        self.settings = self.meta["settings"]
        self.chains = {key: np.load(self.directory / f"chains_{key}.npy", mmap_mode="r")
                       for key in ("A", "Z", "weights", "Gamma", "theta")}
        self.flat = {key: value.reshape(-1, value.shape[-1]) for key, value in self.chains.items()}
        self.isotopes = np.load(self.directory / "isotope_labels.npy").astype(str)
        self.labels = np.load(self.directory / "template_labels.npy").astype(str)
        self.alignments = np.load(self.directory / "alignment_labels.npy").astype(str)
        self.energy = np.load(self.directory / "energy_keV.npy")
        self.observed = np.load(self.directory / "observed_counts.npy")
        self.limits = (self.settings["energy_min"], self.settings["energy_max"])
        if self.energy.min() < self.limits[0] or self.energy.max() > self.limits[1]:
            raise ValueError("Saved fitted energy grid is inconsistent with metadata.")
        self.grid = np.load(self.directory / "interpolation_energy_keV.npy")
        self.templates = np.load(self.directory / "interpolation_isotope_templates.npy")
        self.backgrounds = np.load(self.directory / "interpolation_background_templates.npy")
        self.groups = [np.flatnonzero([label.split("_", 1)[0] == iso for label in self.labels])
                       for iso in self.isotopes]
        self.pip = self.flat["Z"].mean(axis=0)
        self.selected = np.flatnonzero(self.pip >= .5)
        self.clusters = pd.read_csv(self.directory / "template_clusters.csv")
        self.names = template_names(self.clusters.template)


# Rebuild only the context spectra for Figures 1, 2 and 4, using the SAME
# constrained-MLE preprocessing as the sampler. No likelihood fitting is done.
def load_context(p, data_dir):
    from gamma_spec_bayes import Dataset, Settings, pooled_background, register
    dataset = Dataset(data_dir, Settings(**p.settings))
    candidate = {}
    for _, label, runs in dataset.templates:
        times = np.array([s.livetime for s, _ in runs])
        spectra = np.array([register(response, s.calibration, p.grid) for s, response in runs])
        candidate[label] = np.average(spectra, axis=0, weights=times)
    # Catch changed source data before depicting a different library from the fit.
    np.testing.assert_allclose(np.column_stack([candidate[label] for label in p.labels]),
                               p.templates, rtol=1e-10, atol=1e-14)
    background = np.column_stack([pooled_background(dataset.backgrounds[series], p.grid)
                                   for series in p.meta["background_series"]])
    return candidate, background


# Figure 1: one mean candidate-template response for each of the 11 isotopes.
def plot_library(p, candidate, output, manifest):
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.6))
    mask = (p.grid >= p.limits[0]) & (p.grid <= p.limits[1])
    for isotope in p.isotopes:
        response = np.mean([v for label, v in candidate.items() if label.split("_", 1)[0] == isotope], axis=0)
        ax.step(p.grid[mask], np.ma.masked_less_equal(response[mask], 0), where="mid", alpha=.9, label=isotope)
    ax.set(xlim=p.limits, yscale="log", xlabel="Energy (keV)", ylabel="Template response (cps/Bq)")
    ax.legend(ncol=4, frameon=False, loc="lower center")
    save_figure(fig, output, "fig01_isotope_templates",
        "Mean background-corrected candidate-template response for each isotope, over 20--2800 keV. "
        "Zero MLE channels are omitted on the logarithmic axis.", manifest)


# Figure 2: both measured background series before representative selection.
@plt.rc_context(SMALL_FONTS)
def plot_backgrounds(p, background, output, manifest):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    mask = (p.grid >= p.limits[0]) & (p.grid <= p.limits[1])
    for b in range(background.shape[1]):
        ax.plot(p.grid[mask], np.ma.masked_less_equal(background[mask, b], 0), label=rf"$b_{{{b}}}$")
    ax.set(xlim=p.limits, yscale="log", xlabel="Energy (keV)", ylabel="Background count rate (cps)")
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, output, "fig02_background_components",
        "Livetime-weighted empirical background spectra before clustering, over 20--2800 keV. "
        + "; ".join(f"b{i}: {name}" for i, name in enumerate(p.meta["background_series"])), manifest)


# Figure 3: the saved, pre-clustering cosine similarity matrix (same color scale).
def plot_similarity(p, output, manifest):
    matrix = pd.read_csv(p.directory / "template_cosine_similarity.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.7))
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=.95, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(matrix)), [p.names[x] for x in matrix.index], rotation=90)
    ax.set_yticks(range(len(matrix)), [p.names[x] for x in matrix.index])
    fig.colorbar(im, ax=ax, fraction=.046, pad=.025).set_label("Cosine similarity")
    save_figure(fig, output, "fig03_template_cosine_similarity",
        "Pairwise cosine similarity before clustering, evaluated on the fitted 20--2800 keV grid. "
        "The displayed color range remains 0.95--1.00; lower similarities saturate at the lower color.", manifest)


# Figure 4: all candidate templates for the selected isotopes, before clustering.
@plt.rc_context({"legend.fontsize": 8})
def plot_candidate_panels(p, candidate, output, manifest):
    if not len(p.selected):
        return
    # Preserve the manuscript's Co57/Cd109/Am241/Pb210 panel order when available.
    order = [x for x in ("Co57", "Cd109", "Am241", "Pb210") if x in p.isotopes[p.selected]]
    order += [x for x in p.isotopes[p.selected] if x not in order]
    fig, axes = panels(len(order), height=2.8)
    locations = {"Co57": "lower left", "Cd109": "upper left", "Am241": "center right", "Pb210": "lower left"}
    mask = (p.grid >= 20) & (p.grid <= 300)
    for ax, isotope in zip(axes, order):
        for label, values in candidate.items():
            if label.split("_", 1)[0] == isotope:
                ax.plot(p.grid[mask], np.ma.masked_less_equal(values[mask], 0), label=p.names[label])
        panel_label(ax, isotope)
        ax.set(xlim=(20, 300), yscale="log", xlabel="Energy (keV)", ylabel="Template response (cps/Bq)")
        if locations.get(isotope) == "lower left":
            low, high = ax.get_ylim()
            ax.set_ylim(bottom=low / (high/low)**.22)  # Space for the original in-panel legend.
        elif locations.get(isotope) == "upper left":
            low, high = ax.get_ylim()
            ax.set_ylim(top=high * (high/low)**.18)
        anchor = (1, .70) if locations.get(isotope) == "center right" else None
        ax.legend(loc=locations.get(isotope, "best"), bbox_to_anchor=anchor,
                  frameon=False, ncol=2 if isotope == "Co57" else 1)
    save_figure(fig, output, "fig04_candidate_templates",
        "Background-corrected candidate templates for the selected isotopes, before clustering, over 20--300 keV. "
        "Roman numerals follow original candidate order, including candidates later removed.", manifest)


# Figure 5: Co-57 illustrates its actual isotope-specific spike-and-slab prior.
@plt.rc_context(SMALL_FONTS)
def plot_prior(p, output, manifest):
    prior = pd.read_csv(p.directory / "slab_priors.csv").set_index("isotope").loc["Co57"]
    alpha, beta = p.settings["slab_shape"], prior.slab_scale_Bq
    pi = p.settings["inclusion_probability"]
    xmax = invgamma.ppf(.99, a=alpha, scale=beta) / 1000
    x = np.linspace(xmax / 10000, xmax, 2000)
    density = pi * invgamma.pdf(x, a=alpha, scale=beta / 1000)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.plot(x, density, color="black", lw=1)
    ax.annotate("", xy=(0, density.max() * 1.12), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=1))
    ax.text(.95, .94, f"Co57\n$\\alpha={alpha:g}$\n$\\beta={beta:.3f}$ Bq",
            transform=ax.transAxes, ha="right", va="top")
    ax.set(xlim=(-.06*xmax, xmax), ylim=(0, 1.2*density.max()),
           xlabel=r"Co-57 activity $A_j$ (kBq)", ylabel=r"Density (kBq$^{-1}$)")
    save_figure(fig, output, "fig05_activity_prior_Co57",
        f"Co-57 example: IG({alpha:g}, {beta:.6g} Bq), inclusion probability {pi:g}. "
        f"Black curve is the weighted slab density; blue arrow indicates spike mass {1-pi:g}. "
        "Arrow height is symbolic, not a density value. Other isotopes have different slab scales.", manifest)


# Figure 6: activity traces for PIP-selected isotopes, using retained draws only.
@plt.rc_context(SMALL_FONTS)
def plot_traces(p, output, manifest):
    if not len(p.selected):
        return
    fig, axes = plt.subplots(len(p.selected), 1, squeeze=False,
                             figsize=(SINGLE, 1.2 * len(p.selected) + .3), sharex=True)
    n_draws = p.chains["A"].shape[1]
    for ax, j in zip(axes.ravel(), p.selected):
        for chain in range(p.chains["A"].shape[0]):
            ax.plot(np.arange(1, n_draws + 1), p.chains["A"][chain, :, j] / 1000,
                    lw=.35, alpha=.8, rasterized=True)
        ax.set_ylabel(f"{p.isotopes[j]}\n(kBq)")
        ax.yaxis.set_major_locator(MaxNLocator(3))
    axes[-1, 0].set_xlabel("Post-burn-in draw")
    save_figure(fig, output, "fig06_trace_activities",
        "Retained activity traces; colors denote separate chains. No additional burn-in is removed.", manifest)


# Figures 7 and 9: filled marginal densities, black mean and dashed 95% bounds.
def density_panel(ax, values, color, nonnegative=False):
    values = np.asarray(values)
    low, high = np.quantile(values, [.025, .975])
    zero_mass = np.mean(values == 0) if nonnegative else 0
    continuous = values[values > 0] if zero_mass else values
    if len(continuous) > 1 and np.ptp(continuous) > 0:
        a, b = np.quantile(continuous, [.001, .999])
        padding = .1 * (b - a)
        x = np.linspace(max(0, a-padding) if nonnegative else a-padding, b+padding, 400)
        density = (1-zero_mass) * gaussian_kde(continuous)(x)
        ax.plot(x, density, color=color)
        ax.fill_between(x, density, color=color, alpha=.75)
    if zero_mass:
        ax.text(.03, .92, f"P(A=0)={zero_mass:.3g}", transform=ax.transAxes, va="top")
    ax.axvline(values.mean(), color="black", lw=.9)
    ax.axvline(low, color="gray", ls="--", lw=.7)
    ax.axvline(high, color="gray", ls="--", lw=.7)
    ax.set_ylabel("Density")
    # Keep the original plotting.py automatic tick spacing for these densities.
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def plot_marginals(p, output, manifest):
    if not len(p.selected):
        return
    for key, color, scale, xlabel, name in [
        ("A", "steelblue", .001, r"Activity $A_j$ (kBq)", "fig07_posterior_activities"),
        ("theta", "darkorange", 1, r"Shift $\theta_j$ (keV)", "fig09_posterior_isotope_shifts"),
    ]:
        fig, axes = panels(len(p.selected))
        for ax, j in zip(axes, p.selected):
            density_panel(ax, p.flat[key][:, j] * scale, color, nonnegative=(key == "A"))
            ax.set_xlabel(xlabel)
            panel_label(ax, p.isotopes[j])
        save_figure(fig, output, name,
            "Marginal posterior densities for selected isotopes. Solid black: mean; dashed gray: 95% equal-tailed bounds. "
            "Activity densities show the continuous component; any atom at zero is annotated separately.", manifest)


# Figure 8: pairwise activity contours with the original 95/80/50% region legend.
def plot_contours(p, output, manifest):
    pairs = list(combinations(p.selected, 2))
    if not pairs:
        return
    fig, axes = panels(len(pairs), height=2.5, columns=3)
    colors = ["#d9e8f5", "#91bad8", "#3c78a8"]
    edges = ["#6c9fc3", "#3978a8", "#174f7a"]
    for ax, (j, k) in zip(axes, pairs):
        included = (p.flat["Z"][:, j] == 1) & (p.flat["Z"][:, k] == 1)
        x, y = p.flat["A"][included, j]/1000, p.flat["A"][included, k]/1000
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            ax.text(.5, .5, "Insufficient variation", ha="center", transform=ax.transAxes)
            continue
        ranges = []
        for values in (x, y):
            lo, hi = np.quantile(values, [.001, .999])
            ranges.append((lo-.08*(hi-lo), hi+.08*(hi-lo)))
        hist, xe, ye = np.histogram2d(x, y, bins=120, range=ranges)
        density = gaussian_filter(hist.astype(float), sigma=2)
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered) / ordered.sum()
        thresholds = ordered[np.searchsorted(cumulative, [.95, .8, .5])]
        if len(np.unique(thresholds)) == 3:
            xc, yc = (xe[1:]+xe[:-1])/2, (ye[1:]+ye[:-1])/2
            ax.contourf(xc, yc, density.T, levels=np.r_[thresholds, density.max()*(1+1e-9)], colors=colors)
            ax.contour(xc, yc, density.T, levels=thresholds, colors=edges, linewidths=.6)
        ax.plot(x.mean(), y.mean(), "+", color="black", ms=6)
        note = rf"$\rho={np.corrcoef(x, y)[0, 1]:.2f}$"
        if not included.all():
            note += f"\nP(both)={included.mean():.3f}"
        panel_label(ax, note)
        ax.set(xlim=ranges[0], ylim=ranges[1], xlabel=f"{p.isotopes[j]} activity (kBq)",
               ylabel=f"{p.isotopes[k]} activity (kBq)")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(style="plain", useOffset=False)
    handles = [Patch(facecolor=c, edgecolor=e, label=f"{pct}% region")
               for c, e, pct in zip(colors, edges, (95, 80, 50))]
    handles.append(Line2D([], [], marker="+", ls="none", color="black", label="Posterior mean"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .005))
    save_figure(fig, output, "fig08_activity_bivariate_contours",
        "Pairwise activity posteriors: approximate 95%, 80% and 50% highest-density regions from smoothed histograms. "
        "Contours and correlations condition on both isotopes being included; any nonunit joint PIP is annotated.",
        manifest, rect=(0, .065, 1, 1))


# Figures 10 and 11: average Eq. (10) over ALL retained draws, including shifts
# and weights. The sampler's quick 400-draw preview is not used for paper plots.
def reconstruct_posterior(p, batch_size=64):
    functions = [PchipInterpolator(p.grid, v, extrapolate=True) for v in p.templates.T]
    bg_functions = [PchipInterpolator(p.grid, v, extrapolate=True) for v in p.backgrounds.T]
    n = len(p.flat["A"])
    total = np.zeros(len(p.energy))
    components = np.zeros((len(p.isotopes), len(p.energy)))
    background = np.zeros(len(p.energy))
    time = p.meta["livetime_s"]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        rows = slice(start, stop)
        batch = np.zeros((stop-start, len(p.energy)))
        for j, group in enumerate(p.groups):
            active = np.flatnonzero(p.flat["Z"][rows, j])
            if not len(active):
                continue
            draws = start + active
            shifted = p.energy[None, :] + p.flat["theta"][draws, j, None]
            response = sum(p.flat["weights"][draws, m, None] * functions[m](shifted) for m in group)
            rate = p.flat["A"][draws, j, None] * response
            batch[active] += rate
            components[j] += rate.sum(axis=0)
        for b, f in enumerate(bg_functions):
            rate = p.flat["Gamma"][rows, b, None] * f(p.energy[None, :] + p.flat["theta"][rows, len(p.isotopes)+b, None])
            batch += rate
            background += rate.sum(axis=0)
        total += np.maximum(batch, 1e-12 / time).sum(axis=0)
        if stop % (batch_size * 100) == 0 or stop == n:
            print(f"  Reconstructed {stop}/{n} retained draws", flush=True)
    return dict(energy_keV=p.energy, observed_rate=p.observed/time,
                posterior_mean_rate=total/n, isotope_rates=components/n, background_rate=background/n)


def plot_reconstruction(p, reconstruction, output, manifest):
    energy, observed, fitted = (reconstruction[k] for k in ("energy_keV", "observed_rate", "posterior_mean_rate"))
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.1))
    ax.step(energy, np.ma.masked_less_equal(observed, 0), where="mid", alpha=.75, label="Observed")
    ax.step(energy, fitted, where="mid", alpha=.75, label="Posterior mean")
    # Figure 10 display crop; reconstruction still uses the complete fitted grid.
    ax.set(xlim=(20, 1500), yscale="log", xlabel="Energy (keV)", ylabel=r"Count rate (s$^{-1}$)")
    ax.set_ylim(bottom=1e-4)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output, "fig10_spectral_reconstruction",
        "Observed count-rate spectrum and posterior mean reconstruction displayed over 20--1500 keV, "
        "with the lower y-axis limit at 1e-4 counts/s. The fitted range remains 20--2800 keV. "
        "Mean reconstruction uses every retained draw. Zero observed counts are omitted on the log axis.", manifest)
    if not len(p.selected):
        return
    fig, axes = panels(len(p.selected), height=2.3)
    for ax, j in zip(axes, p.selected):
        lo, hi = PEAK_WINDOWS[p.isotopes[j]]
        mask = (energy >= lo) & (energy <= hi)
        ax.step(energy[mask], np.ma.masked_less_equal(observed[mask], 0), where="mid")
        ax.step(energy[mask], fitted[mask], where="mid")
        ax.set(xlim=(lo, hi), yscale="log", xlabel="Energy (keV)", ylabel=r"Count rate (s$^{-1}$)")
        panel_label(ax, p.isotopes[j])
    save_figure(fig, output, "fig11_zoomed_peaks",
        "Selected photopeak windows. Blue: observed; orange: posterior mean, as in Figure 10.", manifest)


# Table summaries retain full numerical precision for CSV export.
def summary(values):
    lo, hi = np.quantile(values, [.005, .995])
    return dict(mean=float(np.mean(values)), low=float(lo), high=float(hi))


# Table I: saved rank-normalized R-hat, bulk ESS and tail ESS for selected A/theta.
def table_diagnostics(p, output):
    diagnostics = pd.read_csv(p.directory / "parameter_diagnostics.csv").set_index("parameter")
    raw = []
    for j in p.selected:
        for key in ("A", "theta"):
            name = f"{key}_{p.isotopes[j]}"
            row = diagnostics.loc[name]
            raw.append(dict(parameter=name, **row[["Rhat", "ESS_bulk", "ESS_tail"]].to_dict()))
    pd.DataFrame(raw).to_csv(output / "table01_convergence.csv", index=False)


# Table II: unconditional activity means/99% intervals and PIPs for all isotopes.
def table_activities(p, output):
    raw = []
    for j, name in enumerate(p.isotopes):
        row = summary(p.flat["A"][:, j] / 1000)
        raw.append(dict(isotope=name, mean_kBq=row["mean"], CI99_low_kBq=row["low"],
                        CI99_high_kBq=row["high"], PIP=p.pip[j]))
    pd.DataFrame(raw).to_csv(output / "table02_activities.csv", index=False)


# Table III: selected isotope shifts and every retained background shift, in keV.
def table_shifts(p, output):
    raw = []
    indices = list(p.selected) + list(range(len(p.isotopes), len(p.alignments)))
    for j in indices:
        row = summary(p.flat["theta"][:, j])
        raw.append(dict(component=p.alignments[j], mean_keV=row["mean"], CI99_low_keV=row["low"], CI99_high_keV=row["high"]))
    pd.DataFrame(raw).to_csv(output / "table03_energy_shifts.csv", index=False)


# Table IV: selected-isotope template weights; names retain pre-clustering IDs.
def table_weights(p, output):
    raw = []
    for j in p.selected:
        for m in p.groups[j]:
            row = summary(p.flat["weights"][:, m])
            raw.append(dict(isotope=p.isotopes[j], template=p.labels[m], display_template=p.names[p.labels[m]],
                            mean=row["mean"], CI99_low=row["low"], CI99_high=row["high"]))
    pd.DataFrame(raw).to_csv(output / "table04_template_weights.csv", index=False)


# Table V: retained background scaling coefficients, including 99% intervals.
def table_background(p, output):
    raw = []
    for b in range(p.flat["Gamma"].shape[1]):
        row = summary(p.flat["Gamma"][:, b])
        raw.append(dict(component=p.alignments[len(p.isotopes)+b], mean=row["mean"],
                        CI99_low=row["low"], CI99_high=row["high"]))
    pd.DataFrame(raw).to_csv(output / "table05_background_scaling.csv", index=False)


# Table VI and aggregate metrics: recompute bias/E_n from all draws using the
# saved G8 reference activities, uncertainties and decay factors. E_n uses 95%.
def reference_metrics(directory, isotopes, activity):
    reference = pd.read_csv(directory / "reference_comparison_G8.csv").set_index("isotope")
    rows = []
    for j, isotope in enumerate(isotopes):
        if isotope not in reference.index:
            continue
        ref = reference.loc[isotope]
        values = activity[:, j] * ref.decay_factor
        lo, hi = np.quantile(values, [.025, .975])
        mean = values.mean()
        uncertainty = (hi-lo)/2
        rows.append(dict(isotope=isotope, mean_at_reference_Bq=mean, reference_Bq=ref.reference_Bq,
            reference_standard_uncertainty_Bq=ref.reference_standard_uncertainty_Bq,
            posterior_95_halfwidth_Bq=uncertainty, reference_date=ref.reference_date,
            bias_percent=100*(mean/ref.reference_Bq-1),
            En=(mean-ref.reference_Bq)/np.hypot(uncertainty, 2*ref.reference_standard_uncertainty_Bq)))
    return pd.DataFrame(rows)


def table_references(p, output):
    metrics = reference_metrics(p.directory, p.isotopes, p.flat["A"])
    metrics.to_csv(output / "table06_reference_comparison.csv", index=False)


# Aggregate results ONLY: signed relative bias, signed E_n and isotope selection.
# Each reference-isotope/spectrum pair has equal weight, including missed isotopes.
# Duplicate spectra are rejected rather than counted twice across experiment runs.
def aggregate_results(root, output):
    directories = ([root] if (root / "chains_A.npy").exists()
                   else sorted(path.parent for path in root.rglob("chains_A.npy") if (path.parent / "metadata.json").exists()))
    if not directories:
        raise ValueError(f"No updated-sampler results found under {root}")
    output.mkdir(parents=True, exist_ok=True)
    seen, all_metrics, decisions, flags, models = set(), [], [], [], []
    for directory in directories:
        meta = json.loads((directory / "metadata.json").read_text())
        identity = (meta["measurement_series"], meta["filename"])
        if identity in seen:
            raise ValueError(f"Duplicate spectrum {identity}; choose one analysis batch for aggregation.")
        seen.add(identity)
        models.append({k: meta["settings"][k] for k in ("energy_min", "energy_max", "slab_shape", "slab_quantile",
            "detection_alpha", "detection_beta", "inclusion_probability", "correction_background",
            "isotope_similarity", "background_similarity", "shift_prior_sd", "background_prior_sd")})
        if models[-1] != models[0]:
            raise ValueError("The aggregate batch mixes different analysis ranges or prior specifications.")
        isotopes = np.load(directory / "isotope_labels.npy").astype(str)
        a = np.load(directory / "chains_A.npy", mmap_mode="r").reshape(-1, len(isotopes))
        z = np.load(directory / "chains_Z.npy", mmap_mode="r").reshape(-1, len(isotopes))
        metrics = reference_metrics(directory, isotopes, a)
        metrics.insert(0, "filename", meta["filename"])
        metrics.insert(0, "series", meta["measurement_series"])
        all_metrics.append(metrics[["series", "filename", "isotope", "bias_percent", "En"]])
        truth = set(metrics.isotope)
        for isotope, pip in zip(isotopes, z.mean(axis=0)):
            decisions.append(dict(series=meta["measurement_series"], filename=meta["filename"],
                isotope=isotope, PIP=pip, expected=isotope in truth, selected=bool(pip >= .5)))
        if meta["diagnostic_flags"]:
            flags.append(dict(filename=meta["filename"], parameters=meta["diagnostic_flags"]))
    values, selection = pd.concat(all_metrics, ignore_index=True), pd.DataFrame(decisions)
    values.to_csv(output / "aggregate_bias_En_by_spectrum_isotope.csv", index=False)
    selection.to_csv(output / "aggregate_selection_by_spectrum_isotope.csv", index=False)
    summary_rows = []
    for series in [*values.series.unique(), "Overall"]:
        v = values if series == "Overall" else values.loc[values.series == series]
        s = selection if series == "Overall" else selection.loc[selection.series == series]
        exact = s.assign(correct=s.expected == s.selected).groupby(["series", "filename"]).correct.all()
        summary_rows.append(dict(series=series, spectra=len(exact), reference_activity_estimates=len(v),
            mean_relative_bias_percent=v.bias_percent.mean(), mean_En=v.En.mean(), sd_En=v.En.std(ddof=1),
            exact_selection_spectra=int(exact.sum()),
            true_positives=int((s.expected & s.selected).sum()),
            false_negatives=int((s.expected & ~s.selected).sum()),
            false_positives=int((~s.expected & s.selected).sum()),
            true_negatives=int((~s.expected & ~s.selected).sum())))
    pd.DataFrame(summary_rows).to_csv(output / "aggregate_summary.csv", index=False)
    overall = summary_rows[-1]
    (output / "aggregate_methods.txt").write_text(
        f"Included {overall['spectra']} distinct spectra and {overall['reference_activity_estimates']} reference activities.\n"
        "Means are signed, not mean absolute errors. Each reference-isotope/spectrum pair has equal weight.\n"
        "All reference activities enter bias/E_n, including any missed isotope. Extra selected isotopes enter FP only.\n"
        "Expected isotope sets come from saved G8 references. Exact selection requires the whole set to match.\n"
        "E_n uses the 95% posterior interval half-width and twice the reference standard uncertainty.\n"
        f"Spectra with sampler diagnostic flags: {json.dumps(flags)}\n")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    if flags:
        print("Aggregate summaries are provisional: some source analyses have diagnostic flags.", flush=True)


# Optional non-manuscript plots from old plotting.py: background densities,
# isotope decomposition, Cd109 shift overlay and the retained-template library.
def plot_extras(p, reconstruction, output, manifest):
    fig, axes = panels(p.backgrounds.shape[1])
    for b, ax in enumerate(axes):
        density_panel(ax, p.flat["Gamma"][:, b], "seagreen", nonnegative=True)
        ax.set_xlabel(rf"$\gamma_{{{b+1}}}$")
    save_figure(fig, output, "extra_posterior_backgrounds", "Background scale posteriors; mean and 95% bounds.", manifest)
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 4.8), sharex=True)
    mask = (p.energy >= 30) & (p.energy <= 300)
    axes[0].step(p.energy[mask], np.ma.masked_less_equal(reconstruction["observed_rate"][mask], 0),
                 where="mid", color="black", alpha=.75, label="Observed")
    axes[0].plot(p.energy[mask], reconstruction["posterior_mean_rate"][mask], color="tab:red", label="Posterior total")
    axes[0].legend(frameon=False)
    for color, j in zip(plt.cm.tab20(np.linspace(0, 1, max(1, len(p.selected)))), p.selected):
        axes[1].plot(p.energy[mask], np.ma.masked_less_equal(reconstruction["isotope_rates"][j, mask], 0),
                     color=color, label=f"{p.isotopes[j]} (PIP={p.pip[j]:.3f})")
    for ax in axes:
        ax.set(yscale="log", ylabel=r"Count rate (s$^{-1}$)", xlim=(30, 300))
    axes[1].set_xlabel("Energy (keV)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save_figure(fig, output, "extra_isotope_decomposition", "Selected isotope contributions over 30--300 keV.", manifest)
    j = list(p.isotopes).index("Cd109")
    if j in p.selected:
        m = p.groups[j][0]
        shift = p.flat["theta"][:, j].mean()
        # Same E + theta sign convention as the likelihood (old overlay used E - theta).
        shifted = PchipInterpolator(p.grid, p.templates[:, m])(p.energy + shift)
        mask = (p.energy >= 20) & (p.energy <= 300)
        unshifted = PchipInterpolator(p.grid, p.templates[:, m])(p.energy)
        fig, ax = plt.subplots(figsize=(DOUBLE, 3))
        ax.plot(p.energy[mask], np.ma.masked_less_equal(unshifted[mask], 0), label=f"{p.names[p.labels[m]]} unshifted")
        ax.plot(p.energy[mask], np.ma.masked_less_equal(shifted[mask], 0),
                label=rf"Shifted ($\bar\theta={shift:.4f}$ keV)")
        ax.set(xlim=(20, 300), yscale="log", xlabel="Energy (keV)", ylabel="Template response (cps/Bq)")
        ax.legend(frameon=False)
        save_figure(fig, output, "extra_template_Cd109_shift", "Cd109 template evaluated at E + posterior mean shift.", manifest)
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.4))
    mask = (p.grid >= p.limits[0]) & (p.grid <= p.limits[1])
    for m, label in enumerate(p.labels):
        ax.step(p.grid[mask], np.ma.masked_less_equal(p.templates[mask, m], 0), where="mid", alpha=.85, label=p.names[label])
    ax.set(xlim=p.limits, yscale="log", xlabel="Energy (keV)", ylabel="Template response (cps/Bq)")
    ax.legend(ncol=3, frameon=False)
    save_figure(fig, output, "extra_representative_library", "All retained background-corrected templates over 20--2800 keV.", manifest)


# Write figure placement/caption guidance and generate only the requested scope.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, help="One spectrum's updated-sampler output directory")
    parser.add_argument("--aggregate-root", type=Path, help="One batch containing saved results for aggregation")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, help="Default: <results-dir>/manuscript or <aggregate-root>/manuscript")
    parser.add_argument("--skip-context", action="store_true", help="Skip Figures 1, 2 and 4 (no input tables needed)")
    parser.add_argument("--extra-plots", action="store_true", help="Also generate the old non-manuscript plots")
    args = parser.parse_args()
    if args.results_dir is None and args.aggregate_root is None:
        parser.error("Provide --results-dir and/or --aggregate-root.")
    output = args.output_dir or (args.results_dir or args.aggregate_root) / "manuscript"
    output.mkdir(parents=True, exist_ok=True)
    if args.results_dir:
        p, manifest = Posterior(args.results_dir), []
        print(f"Figures/tables for {p.meta['filename']}; fitted range {p.limits[0]:g}--{p.limits[1]:g} keV", flush=True)
        for table in (table_diagnostics, table_activities, table_shifts, table_weights, table_background, table_references):
            table(p, output)
        if not args.skip_context:
            candidate, background = load_context(p, args.data_dir)
            plot_library(p, candidate, output, manifest)
            plot_backgrounds(p, background, output, manifest)
            plot_candidate_panels(p, candidate, output, manifest)
        plot_similarity(p, output, manifest)
        plot_prior(p, output, manifest)
        plot_traces(p, output, manifest)
        plot_marginals(p, output, manifest)
        plot_contours(p, output, manifest)
        reconstruction = reconstruct_posterior(p)
        np.savez(output / "posterior_reconstruction.npz", **reconstruction)
        plot_reconstruction(p, reconstruction, output, manifest)
        if args.extra_plots:
            plot_extras(p, reconstruction, output, manifest)
        (output / "figure_manifest.json").write_text(json.dumps(sorted(manifest, key=lambda x: x["file"]), indent=2) + "\n")
        (output / "READ_ME.txt").write_text(
            "Original panel arrangements, figure dimensions and legend positions are restored.\n"
            "Font sizes follow the original plotting.py defaults: 12 pt text and 10.5 pt legends.\n"
            "Export font sizes are before any resizing in the manuscript.\n"
            "PNG previews are 300 dpi; use PDF for the manuscript. Captions/widths are in figure_manifest.json.\n"
            "Tables are saved only as full-precision CSV files.\n"
            "Tables II--V: 99% equal-tailed intervals; density markers and E_n: 95% intervals.\n"
            "Activity summaries include excluded draws (A=0); [0,0] does not establish physical absence.\n"
            "Figure 1 follows the old plotting code: mean across candidate templates, not a single calibration run.\n"
            "Figure 5 is the Co-57 prior example; its scale is read from this analysis, not fixed at 1500 Bq.\n"
            "Reconstruction averages every retained draw using saved templates and E + theta shifts.\n"
            f"Source: {args.results_dir.resolve()}\n"
            f"Sampler diagnostic flags: {p.meta['diagnostic_flags']}\n")
        if p.meta["diagnostic_flags"]:
            print("Source sampler has diagnostic flags; treat summaries as provisional.", flush=True)
    if args.aggregate_root:
        aggregate_results(args.aggregate_root, output / "aggregate")
    print(f"Saved plots and tables to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
