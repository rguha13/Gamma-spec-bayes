"""Generate manuscript figures from saved Bayesian posterior chains.

This script never runs the MCMC sampler. It reloads the selected observed
spectrum, its fixed energy calibration, the saved representative templates,
and the retained posterior draws written by gamma_spec_bayes.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde, invgamma


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
})


def save_figure(fig, path, dpi=200, show=False):
    """Save and close one figure, optionally displaying it first."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def register_to_energy_grid(values, calibration, channels, common_energy):
    """Register a native-channel spectrum on the selected run's grid."""
    b0, b1 = float(calibration[0]), float(calibration[1])
    if not np.isfinite(b0) or not np.isfinite(b1) or b1 <= 0:
        raise ValueError(f"Invalid energy calibration: b0={b0}, b1={b1}")
    native_energy = b0 + b1 * channels
    registered = PchipInterpolator(
        native_energy, values, extrapolate=True
    )(common_energy)
    target_bin_width = float(np.median(np.diff(common_energy)))
    registered *= target_bin_width / b1
    return np.clip(registered, 0.0, None)


def load_analysis_inputs(base, mixture_index, run_index):
    """Reload the observed spectrum and deterministic calibration inputs."""
    mixture_reference = pd.read_excel(base / "02_A_REF.xlsx")
    template_reference = pd.read_excel(base / "03_Templates_A_REF.xlsx")
    calibration = pd.read_excel(base / "042_DATA_ECal.xlsx")
    spectra = pd.read_excel(base / "04_DATA_spectra.xlsx")

    spectrum_filenames = spectra["Filename"].astype(str).to_numpy()
    calibration_filenames = calibration["Filename"].astype(str).to_numpy()
    if pd.Series(calibration_filenames).duplicated().any():
        raise ValueError("042_DATA_ECal.xlsx contains duplicate filenames.")

    mixture_ids = (
        mixture_reference["MeasurementSeriesID_613"].astype(str).unique()
    )
    if mixture_index < 0 or mixture_index >= len(mixture_ids):
        raise IndexError(
            f"Mixture index {mixture_index} is outside 0..{len(mixture_ids)-1}."
        )
    mixture_id = mixture_ids[mixture_index]
    mixture_rows = mixture_reference.loc[
        mixture_reference["MeasurementSeriesID_613"].astype(str).eq(mixture_id)
    ]
    run_filenames = np.unique(mixture_rows["Filename"].astype(str))
    if run_index < 0 or run_index >= len(run_filenames):
        raise IndexError(
            f"Run index {run_index} is outside 0..{len(run_filenames)-1} "
            f"for {mixture_id}."
        )
    run_filename = run_filenames[run_index]
    spectrum_match = np.where(spectrum_filenames == run_filename)[0]
    calibration_match = np.where(calibration_filenames == run_filename)[0]
    if len(spectrum_match) != 1 or len(calibration_match) != 1:
        raise ValueError(f"Could not uniquely locate {run_filename}.")

    spectrum_index = int(spectrum_match[0])
    calibration_index = int(calibration_match[0])
    counts = spectra.iloc[:, 4:].to_numpy(dtype=float)
    observed_counts = counts[spectrum_index]
    live_time = float(spectra.loc[spectrum_index, "t_live_s"])
    run_calibration = calibration.loc[
        calibration_index, ["Ecal_b0_GX", "Ecal_b1_GX"]
    ].to_numpy(dtype=float)
    channels = np.arange(1, observed_counts.size + 1, dtype=float)
    energy = run_calibration[0] + run_calibration[1] * channels

    return {
        "mixture_reference": mixture_reference,
        "template_reference": template_reference,
        "calibration": calibration,
        "spectra": spectra,
        "counts": counts,
        "spectrum_filenames": spectrum_filenames,
        "calibration_filenames": calibration_filenames,
        "mixture_id": mixture_id,
        "run_filename": run_filename,
        "observed_counts": observed_counts,
        "live_time": live_time,
        "run_calibration": run_calibration,
        "channels": channels,
        "energy": energy,
    }


def load_saved_posterior(result_dir):
    """Load all posterior draws and representative-template artifacts."""
    required = [
        "chains_A.npy", "chains_weights.npy", "chains_Gamma.npy",
        "chains_theta.npy", "chains_Z.npy", "chains_loglik.npy",
        "cluster_representative_templates.npy",
        "background_representative_templates.npy", "isotope_labels.npy",
        "template_labels.npy", "alignment_labels.npy",
        "posterior_template_weights.csv",
        "template_cosine_similarity.csv",
        "analysis_metadata.json", "energy_grid.npy",
    ]
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Saved result directory is missing required files: {missing}"
        )

    chains_A = np.load(result_dir / "chains_A.npy")
    chains_weights = np.load(result_dir / "chains_weights.npy")
    chains_gamma = np.load(result_dir / "chains_Gamma.npy")
    chains_theta = np.load(result_dir / "chains_theta.npy")
    chains_Z = np.load(result_dir / "chains_Z.npy")
    chains_loglik = np.load(result_dir / "chains_loglik.npy")
    saved_energy = np.load(result_dir / "energy_grid.npy")
    with (result_dir / "analysis_metadata.json").open(
        encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)
    isotope_labels = np.load(
        result_dir / "isotope_labels.npy", allow_pickle=True
    ).astype(str)
    template_labels = np.load(
        result_dir / "template_labels.npy", allow_pickle=True
    ).astype(str)
    alignment_labels = np.load(
        result_dir / "alignment_labels.npy", allow_pickle=True
    ).astype(str)
    templates = np.load(result_dir / "cluster_representative_templates.npy")
    backgrounds = np.load(result_dir / "background_representative_templates.npy")

    weight_table = pd.read_csv(
        result_dir / "posterior_template_weights.csv"
    ).sort_values("Cluster_index")
    if len(weight_table) != templates.shape[1]:
        raise ValueError("Template-weight metadata does not match saved templates.")
    isotope_lookup = {name: j for j, name in enumerate(isotope_labels)}
    isotope_of_template = np.array([
        isotope_lookup[name] for name in weight_table["Isotope"].astype(str)
    ], dtype=int)
    isotope_template_indices = [
        np.where(isotope_of_template == j)[0]
        for j in range(len(isotope_labels))
    ]

    return {
        "chains_A": chains_A,
        "chains_weights": chains_weights,
        "chains_gamma": chains_gamma,
        "chains_theta": chains_theta,
        "chains_Z": chains_Z,
        "chains_loglik": chains_loglik,
        "isotope_labels": isotope_labels,
        "template_labels": template_labels,
        "alignment_labels": alignment_labels,
        "templates": templates,
        "backgrounds": backgrounds,
        "isotope_of_template": isotope_of_template,
        "isotope_template_indices": isotope_template_indices,
        "saved_energy": saved_energy,
        "metadata": metadata,
    }


def validate_saved_analysis(data, posterior):
    """Reject saved results that do not match the requested run and model."""
    metadata = posterior["metadata"]
    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported analysis metadata schema: "
            f"{metadata.get('schema_version')!r}."
        )
    try:
        saved_data = metadata["data"]
        model = metadata["model"]
        dimensions = model["dimensions"]
        sampler = metadata["sampler"]
    except KeyError as error:
        raise ValueError(
            f"Analysis metadata is missing required field {error.args[0]!r}."
        ) from error

    identity_checks = {
        "mixture ID": (saved_data.get("mixture_id"), data["mixture_id"]),
        "run filename": (saved_data.get("run_filename"), data["run_filename"]),
    }
    for label, (saved, requested) in identity_checks.items():
        if str(saved) != str(requested):
            raise ValueError(
                f"Saved {label} {saved!r} does not match requested "
                f"{label} {requested!r}."
            )

    saved_calibration = np.array([
        saved_data["energy_calibration"]["b0_keV"],
        saved_data["energy_calibration"]["b1_keV_per_channel"],
    ], dtype=float)
    if not np.allclose(
        saved_calibration, data["run_calibration"], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Saved energy calibration does not match the requested run: "
            f"saved={saved_calibration.tolist()}, "
            f"requested={data['run_calibration'].tolist()}."
        )
    if (
        posterior["saved_energy"].shape != data["energy"].shape
        or not np.allclose(
            posterior["saved_energy"], data["energy"], rtol=0.0, atol=1e-10
        )
    ):
        raise ValueError("Saved energy grid does not match the requested run.")

    K, M, B, P = (
        int(dimensions[name]) for name in ("K", "M", "B", "P")
    )
    if P != K + B:
        raise ValueError("Saved model configuration has inconsistent dimensions.")
    required_model_sections = {"priors", "clustering", "proposals"}
    missing_sections = sorted(required_model_sections - set(model))
    if missing_sections:
        raise ValueError(
            f"Saved model configuration is missing sections: {missing_sections}."
        )
    configured_backgrounds = model.get("background_labels")
    if configured_backgrounds != posterior["alignment_labels"][K:].tolist():
        raise ValueError(
            "Saved background labels do not match the alignment configuration."
        )
    label_dimensions = {
        "isotope_labels": K, "template_labels": M, "alignment_labels": P,
    }
    for name, expected_length in label_dimensions.items():
        if len(posterior[name]) != expected_length:
            raise ValueError(
                f"Saved {name} length {len(posterior[name])} does not match "
                f"model configuration {expected_length}."
            )
    expected_labels = {
        "isotope_labels": posterior["isotope_labels"].tolist(),
        "template_labels": posterior["template_labels"].tolist(),
        "alignment_labels": posterior["alignment_labels"].tolist(),
    }
    for name, artifact_labels in expected_labels.items():
        if model.get(name) != artifact_labels:
            raise ValueError(
                f"Saved model configuration does not match {name}."
            )
    if model.get("isotope_template_indices") != [
        indices.tolist() for indices in posterior["isotope_template_indices"]
    ]:
        raise ValueError(
            "Saved model configuration does not match template-weight metadata."
        )

    n_chains = int(sampler["n_chains"])
    n_draws = int(sampler["retained_draws_per_chain"])
    expected_shapes = {
        "chains_A": (n_chains, n_draws, K),
        "chains_weights": (n_chains, n_draws, M),
        "chains_gamma": (n_chains, n_draws, B),
        "chains_theta": (n_chains, n_draws, P),
        "chains_Z": (n_chains, n_draws, K),
        "chains_loglik": (n_chains, n_draws),
        "templates": (data["energy"].size, M),
        "backgrounds": (data["energy"].size, B),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = posterior[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Saved {name} shape {actual_shape} does not match model "
                f"configuration {expected_shape}."
            )


def plot_activity_prior(output_dir, alpha, beta_bq, pi, show=False):
    """Plot the manuscript spike-and-slab activity prior."""
    activity_kbq = np.linspace(0.001, 12.0, 4000)
    activity_bq = 1000.0 * activity_kbq
    slab_density = (
        pi * invgamma.pdf(activity_bq, a=alpha, scale=beta_bq) * 1000.0
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(activity_kbq, slab_density, color="black", linewidth=1.5)
    spike_top = 1.15 * slab_density.max()
    ax.annotate(
        "", xy=(0.0, spike_top), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.5),
        annotation_clip=False,
    )
    ax.set_xlabel(r"Isotope Activity $A_j$ (kBq)")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, 12)
    ax.set_ylim(0.0, 1.25 * slab_density.max())
    ax.grid(False)
    save_figure(fig, output_dir / "activity_prior.png", dpi=300, show=show)


def build_template_series(measurement_series_id, data):
    """Build one registered cps/Bq template from all valid series runs."""
    rows = data["template_reference"].loc[
        data["template_reference"]["MeasurementSeriesID_613"].astype(str)
        .eq(str(measurement_series_id))
    ]
    if rows.empty:
        raise ValueError(f"No template rows found for {measurement_series_id}.")
    registered_runs, live_times = [], []
    for _, row in rows.iterrows():
        filename = str(row["Filename"])
        spectrum_match = np.where(data["spectrum_filenames"] == filename)[0]
        calibration_match = np.where(
            data["calibration_filenames"] == filename
        )[0]
        if len(spectrum_match) != 1 or len(calibration_match) != 1:
            raise ValueError(f"Could not uniquely locate template run {filename}.")
        spectrum_index = int(spectrum_match[0])
        calibration_index = int(calibration_match[0])
        activity_bq = float(row["A_Bq"])
        live_time = float(data["spectra"].loc[spectrum_index, "t_live_s"])
        if activity_bq <= 0 or live_time <= 0:
            continue
        calibration = data["calibration"].loc[
            calibration_index, ["Ecal_b0_GX", "Ecal_b1_GX"]
        ].to_numpy(dtype=float)
        response = data["counts"][spectrum_index] / live_time / activity_bq
        registered_runs.append(register_to_energy_grid(
            response, calibration, data["channels"], data["energy"]
        ))
        live_times.append(live_time)
    if not registered_runs:
        raise ValueError(f"No valid runs found for {measurement_series_id}.")
    live_times = np.asarray(live_times, dtype=float)
    weights = live_times / live_times.sum()
    return np.sum(weights[:, None] * np.vstack(registered_runs), axis=0)


def plot_original_template_panel(data, output_dir, show=False):
    """Plot all original templates for four example isotope families."""
    isotope_order = ["Co57", "Cd109", "Am241", "Pb210"]
    isotope_display = {
        "Co57": "Co57",
        "Cd109": "Cd109",
        "Am241": "Am241",
        "Pb210": "Pb210",
    }
    legend_locations = {
        "Co57": "lower left", "Cd109": "upper left",
        "Am241": "center right", "Pb210": "lower left",
    }
    roman = ["I", "II", "III", "IV", "V", "VI"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    mask = (data["energy"] >= 20.0) & (data["energy"] <= 300.0)
    for ax, isotope in zip(axes.ravel(), isotope_order):
        series_ids = np.unique(data["template_reference"].loc[
            data["template_reference"]["Nuclide"].astype(str).eq(isotope),
            "MeasurementSeriesID_613",
        ].astype(str))
        if len(series_ids) > len(roman):
            raise ValueError(f"Roman labels are not defined for {isotope}.")
        for local_index, series_id in enumerate(series_ids):
            response = build_template_series(series_id, data)
            ax.plot(
                data["energy"][mask], np.clip(response[mask], 1e-14, None),
                linewidth=1.2,
                label=f"{isotope_display[isotope]}-{roman[local_index]}",
            )
        ax.text(
            0.96, 0.94, isotope_display[isotope], transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )
        ax.set_xlim(20.0, 300.0)
        ax.set_xticks(np.arange(20.0, 301.0, 40.0))
        ax.set_yscale("log")
        if isotope == "Co57":
            ax.set_ylim(bottom=1e-10)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Template response (cps/Bq)")
        ax.legend(
            loc=legend_locations[isotope], frameon=False, fontsize=10,
            ncol=2 if isotope == "Co57" else 1,
        )
        ax.grid(False)
    save_figure(fig, output_dir / "templates_panel.png", dpi=300, show=show)


def build_background_series(measurement_series_id, data):
    """Build one registered empirical background count-rate spectrum."""
    rows = data["spectra"].loc[
        data["spectra"]["MeasurementSeriesID_613"].astype(str)
        .eq(str(measurement_series_id))
    ]
    registered_runs, live_times = [], []
    for spectrum_index, row in rows.iterrows():
        filename = str(row["Filename"])
        calibration_match = np.where(
            data["calibration_filenames"] == filename
        )[0]
        if len(calibration_match) != 1:
            raise ValueError(f"Could not locate background calibration {filename}.")
        live_time = float(row["t_live_s"])
        if live_time <= 0:
            continue
        calibration = data["calibration"].loc[
            int(calibration_match[0]), ["Ecal_b0_GX", "Ecal_b1_GX"]
        ].to_numpy(dtype=float)
        response = data["counts"][spectrum_index] / live_time
        registered_runs.append(register_to_energy_grid(
            response, calibration, data["channels"], data["energy"]
        ))
        live_times.append(live_time)
    if not registered_runs:
        raise ValueError(f"No valid background runs for {measurement_series_id}.")
    live_times = np.asarray(live_times, dtype=float)
    weights = live_times / live_times.sum()
    return np.sum(weights[:, None] * np.vstack(registered_runs), axis=0)


def plot_background_overlay(data, output_dir, show=False):
    """Plot the two original empirical background components."""
    background_mask = (
        data["spectra"]["MeasurementSeriesID_613"].astype(str)
        .str.contains("-NE-", regex=False)
    )
    series_ids = np.unique(data["spectra"].loc[
        background_mask, "MeasurementSeriesID_613"
    ].astype(str))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mask = (data["energy"] >= 20.0) & (data["energy"] <= 1500.0)
    for b, series_id in enumerate(series_ids):
        response = build_background_series(series_id, data)
        ax.plot(
            data["energy"][mask], np.clip(response[mask], 1e-14, None),
            linewidth=1.2, label=rf"$b_{{{b}}}$",
        )
    ax.set_xlim(20.0, 1500.0)
    ax.set_xticks([20.0, 300.0, 600.0, 900.0, 1200.0, 1500.0])
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Background count rate (cps)")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(False)
    save_figure(
        fig, output_dir / "background_components.png", dpi=300, show=show
    )


def plot_template_similarity(result_dir, show=False):
    """Regenerate the cosine-similarity map used for template clustering."""
    similarity_frame = pd.read_csv(
        result_dir / "template_cosine_similarity.csv", index_col=0
    )
    similarity = similarity_frame.to_numpy(dtype=float)
    labels = similarity_frame.index.astype(str).to_numpy()
    roman = ["I", "II", "III", "IV", "V", "VI"]
    isotope_counts = {}
    display_labels = []
    for label in labels:
        isotope = label.split("_", 1)[0]
        isotope_counts[isotope] = isotope_counts.get(isotope, 0) + 1
        display_labels.append(
            f"{isotope}-{roman[isotope_counts[isotope] - 1]}"
        )
    indices = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 7.8))
    image = ax.imshow(
        similarity, cmap="viridis", vmin=0.95, vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(display_labels, rotation=90, fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    save_figure(
        fig, result_dir / "template_cosine_similarity.png", show=show
    )


def reconstruct_posterior(data, posterior, batch_size=100):
    """Average the reconstructed count-rate spectrum over retained draws."""
    energy = data["energy"]
    templates = posterior["templates"]
    backgrounds = posterior["backgrounds"]
    isotope_template_indices = posterior["isotope_template_indices"]
    isotope_labels = posterior["isotope_labels"]
    template_interpolators = [
        PchipInterpolator(energy, templates[:, m], extrapolate=True)
        for m in range(templates.shape[1])
    ]
    background_interpolators = [
        PchipInterpolator(energy, backgrounds[:, b], extrapolate=True)
        for b in range(backgrounds.shape[1])
    ]

    activity = posterior["chains_A"].reshape(-1, len(isotope_labels))
    weights = posterior["chains_weights"].reshape(-1, templates.shape[1])
    gamma = posterior["chains_gamma"].reshape(-1, backgrounds.shape[1])
    theta = posterior["chains_theta"].reshape(
        -1, len(isotope_labels) + backgrounds.shape[1]
    )
    inclusion = posterior["chains_Z"].reshape(-1, len(isotope_labels))
    pip = inclusion.mean(axis=0)

    component_mask = (energy >= 30.0) & (energy <= 300.0)
    total_sum = np.zeros(energy.size, dtype=float)
    component_sum = np.zeros((len(isotope_labels), component_mask.sum()))
    for start in range(0, len(activity), batch_size):
        stop = min(start + batch_size, len(activity))
        rows = slice(start, stop)
        batch_total = np.zeros((stop - start, energy.size), dtype=float)
        for j in range(len(isotope_labels)):
            isotope_response = np.zeros_like(batch_total)
            shifted_energy = energy[None, :] + theta[rows, j, None]
            for m in isotope_template_indices[j]:
                isotope_response += (
                    weights[rows, m, None]
                    * template_interpolators[m](shifted_energy)
                )
            isotope_rate = (
                activity[rows, j, None]
                * inclusion[rows, j, None]
                * isotope_response
            )
            batch_total += isotope_rate
            component_sum[j] += isotope_rate[:, component_mask].sum(axis=0)
        for b in range(backgrounds.shape[1]):
            shifted_energy = (
                energy[None, :] + theta[rows, len(isotope_labels) + b, None]
            )
            batch_total += (
                gamma[rows, b, None]
                * background_interpolators[b](shifted_energy)
            )
        total_sum += batch_total.sum(axis=0)

    return {
        "activity": activity,
        "weights": weights,
        "gamma": gamma,
        "theta": theta,
        "inclusion": inclusion,
        "pip": pip,
        "selected": np.where(pip >= 0.5)[0],
        "rate_observed": data["observed_counts"] / data["live_time"],
        "rate_mean": total_sum / len(activity),
        "component_mask": component_mask,
        "component_rate": component_sum / len(activity),
        "template_interpolators": template_interpolators,
    }


def plot_reconstruction(data, reconstruction, result_dir, show=False):
    """Plot the full posterior-mean reconstruction and mixture peak windows."""
    energy = data["energy"]
    observed = reconstruction["rate_observed"]
    fitted = reconstruction["rate_mean"]
    mask = (energy >= 20.0) & (energy <= 1500.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(energy[mask], observed[mask], where="mid", label="Observed", alpha=0.75)
    ax.step(
        energy[mask], fitted[mask], where="mid", label="Posterior mean",
        alpha=0.75,
    )
    ax.set_xlim(20, 1500)
    ax.set_ylim(1e-5, 1e1)
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"Count rate (s$^{-1}$)")
    ax.legend(frameon=False, fontsize=12)
    save_figure(fig, result_dir / "spectral_reconstruction.png", show=show)

    zoom_regions = {
        0: [(810, 860, "Mn54"), (1145, 1360, "Co60"),
            (1085, 1145, "Zn65"), (635, 685, "Cs137")],
        1: [(100, 150, "Co57"), (65, 115, "Cd109"),
            (27, 56, "Pb210"), (53, 85, "Am241")],
        2: [(810, 860, "Mn54"), (100, 150, "Co57"),
            (1145, 1360, "Co60"), (775, 825, "Cs134"),
            (635, 685, "Cs137"), (53, 85, "Am241")],
        3: [(1245, 1300, "Na22"), (1307, 1357, "Co60"),
            (875, 925, "Y88"), (775, 825, "Cs134")],
    }
    regions = zoom_regions[data["mixture_index"]]
    ncols = 2 if len(regions) <= 4 else 3
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (emin, emax, label) in zip(axes, regions):
        local = (energy >= emin) & (energy <= emax)
        ax.step(energy[local], np.clip(observed[local], 1e-12, None), where="mid")
        ax.step(energy[local], np.clip(fitted[local], 1e-12, None), where="mid")
        ax.text(
            0.96, 0.94, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )
        ax.set_yscale("log")
        ax.set_xlim(emin, emax)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Rate")
    for ax in axes[len(regions):]:
        ax.axis("off")
    save_figure(fig, result_dir / "zoomed_peaks.png", show=show)


def plot_isotope_decomposition(data, posterior, reconstruction, result_dir,
                               show=False):
    """Plot total fit and PIP-selected isotope components from 30--300 keV."""
    mask = reconstruction["component_mask"]
    energy = data["energy"][mask]
    selected = reconstruction["selected"]
    fig, (ax_total, ax_components) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    ax_total.step(
        energy, np.clip(reconstruction["rate_observed"][mask], 1e-12, None),
        where="mid", color="black", linewidth=1.0, alpha=0.75,
        label="Observed",
    )
    ax_total.plot(
        energy, np.clip(reconstruction["rate_mean"][mask], 1e-12, None),
        color="tab:red", linewidth=1.5, label="Posterior total",
    )
    ax_total.set_yscale("log")
    ax_total.set_ylabel(r"Count rate (s$^{-1}$)")
    ax_total.legend(frameon=False)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(selected), 1)))
    for color, j in zip(colors, selected):
        ax_components.plot(
            energy, np.clip(reconstruction["component_rate"][j], 1e-12, None),
            color=color, linewidth=1.2,
            label=(f"{posterior['isotope_labels'][j]} "
                   f"(PIP={reconstruction['pip'][j]:.3f})"),
        )
    ax_components.set_yscale("log")
    ax_components.set_xlim(30.0, 300.0)
    ax_components.set_xlabel("Energy (keV)")
    ax_components.set_ylabel(r"Component rate (s$^{-1}$)")
    ax_components.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5, frameon=False,
    )
    save_figure(
        fig, result_dir / "pip_selected_isotope_decomposition_30_300keV.png",
        show=show,
    )


def plot_filled_density(ax, samples, color, lower_bound=None):
    """Plot a smooth posterior-density curve with filled area."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot plot a density with no finite samples.")
    center = float(values.mean())
    spread = float(values.std())
    minimum_scale = max(abs(center) * 1e-6, 1e-8)
    if values.size < 2 or spread <= minimum_scale:
        scale = max(spread, minimum_scale)
        x = np.linspace(center - 4 * scale, center + 4 * scale, 400)
        density = np.exp(-0.5 * ((x - center) / scale) ** 2)
        density /= scale * np.sqrt(2 * np.pi)
    else:
        kde = gaussian_kde(values)
        lower, upper = np.percentile(values, [0.1, 99.9])
        span = max(float(upper - lower), spread)
        x = np.linspace(lower - 0.1 * span, upper + 0.1 * span, 400)
        density = kde(x)
    if lower_bound is not None:
        keep = x >= lower_bound
        x, density = x[keep], density[keep]
    ax.plot(x, density, color=color, linewidth=1.2)
    ax.fill_between(x, 0.0, density, color=color, alpha=0.75)


def plot_traces_and_posteriors(posterior, reconstruction, result_dir,
                               show=False):
    """Plot activity traces and marginal posterior densities."""
    labels = posterior["isotope_labels"]
    selected = reconstruction["selected"]
    active_labels = labels[selected]
    chains_A = posterior["chains_A"]
    chains_theta = posterior["chains_theta"]
    n_chains, n_draws, _ = chains_A.shape

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(7.16, 2.0 * len(selected)), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, j, label in zip(axes, selected, active_labels):
        for chain in range(n_chains):
            ax.plot(
                np.arange(n_draws), chains_A[chain, :, j] / 1e3,
                linewidth=0.7, alpha=0.8,
            )
        ax.set_ylabel(f"{label}\n(kBq)")
    axes[-1].set_xlabel("Post-burn-in draw")
    save_figure(fig, result_dir / "trace_activities.png", show=show)

    activity = reconstruction["activity"] / 1e3
    ncols = 2 if len(selected) <= 4 else 3
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, j in zip(axes, selected):
        values = activity[:, j]
        mean = values.mean()
        lower, upper = np.percentile(values, [2.5, 97.5])
        plot_filled_density(ax, values, "steelblue", lower_bound=0.0)
        ax.axvline(mean, linewidth=1.0, color="black")
        ax.axvline(lower, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(upper, linestyle="--", linewidth=0.8, color="gray")
        ax.text(
            0.96, 0.94, labels[j], transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )
        ax.set_xlabel(r"Activity $A_j$ (kBq)")
        ax.set_ylabel("Density")
    for ax in axes[len(selected):]:
        ax.axis("off")
    save_figure(fig, result_dir / "posterior_activities.png", show=show)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    theta = reconstruction["theta"]
    for ax, j in zip(axes, selected):
        values = theta[:, j]
        mean = values.mean()
        lower, upper = np.percentile(values, [2.5, 97.5])
        plot_filled_density(ax, values, "darkorange")
        ax.axvline(mean, linewidth=1.0, color="black")
        ax.axvline(lower, linestyle="--", linewidth=0.8, color="gray")
        ax.axvline(upper, linestyle="--", linewidth=0.8, color="gray")
        ax.text(
            0.96, 0.94, labels[j], transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )
        ax.set_xlabel(r"Shift $\theta_j$ (keV)")
        ax.set_ylabel("Density")
    for ax in axes[len(selected):]:
        ax.axis("off")
    save_figure(fig, result_dir / "posterior_isotope_shifts.png", show=show)

    gamma = reconstruction["gamma"]
    n_background = gamma.shape[1]
    bg_cols = min(n_background, 2)
    bg_rows = int(np.ceil(n_background / bg_cols))
    fig, axes = plt.subplots(bg_rows, bg_cols, figsize=(7.16, 1.9 * bg_rows))
    axes = np.atleast_1d(axes).ravel()
    for b in range(n_background):
        values = gamma[:, b]
        mean = values.mean()
        lower, upper = np.percentile(values, [2.5, 97.5])
        plot_filled_density(axes[b], values, "seagreen", lower_bound=0.0)
        axes[b].axvline(mean, linewidth=1.0, color="black")
        axes[b].axvline(lower, linestyle="--", linewidth=0.8, color="gray")
        axes[b].axvline(upper, linestyle="--", linewidth=0.8, color="gray")
        axes[b].set_xlabel(rf"$\gamma_{{{b}}}$")
        axes[b].set_ylabel("Density")
    for ax in axes[n_background:]:
        ax.axis("off")
    save_figure(fig, result_dir / "posterior_backgrounds.png", show=show)


def plot_template_overlays(data, posterior, reconstruction, result_dir,
                           show=False):
    """Plot the Cd109 shift example and complete representative library."""
    labels = posterior["isotope_labels"]
    cd_index = int(np.where(labels == "Cd109")[0][0])
    template_index = int(posterior["isotope_template_indices"][cd_index][0])
    theta_mean = float(reconstruction["theta"][:, cd_index].mean())
    energy = data["energy"]
    mask = (energy >= 20.0) & (energy <= 300.0)
    unshifted = posterior["templates"][:, template_index]
    shifted = reconstruction["template_interpolators"][template_index](
        energy + theta_mean
    )
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    ax.plot(
        energy[mask], np.clip(unshifted[mask], 1e-12, None),
        linewidth=1.2, label=rf"$m={template_index}$ unshifted",
    )
    ax.plot(
        energy[mask], np.clip(shifted[mask], 1e-12, None), linewidth=1.2,
        label=(rf"$m={template_index}$ shifted "
               rf"($\bar{{\theta}}={theta_mean:.4f}$ keV)"),
    )
    ax.set_yscale("log")
    ax.set_xlim(20.0, 300.0)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Template response (cps/Bq)")
    ax.legend(frameon=False)
    ax.text(
        0.97, 0.74, "Cd109", transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )
    save_figure(
        fig, result_dir / "template_Cd109_shift_overlay.png", show=show
    )

    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    for m, label in enumerate(posterior["template_labels"]):
        ax.step(
            energy, np.clip(posterior["templates"][:, m], 1e-12, None),
            where="mid", linewidth=0.9, alpha=0.85, label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Template response (cps/Bq)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    save_figure(
        fig, result_dir / "cluster_representative_template_library.png",
        show=show,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all manuscript plots from saved MCMC chains."
    )
    parser.add_argument("--mixture", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing the Wübbeler et al. data files.",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-library-plots", action="store_true",
        help="Skip the prior, original-template, and background context plots.",
    )
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    base = arguments.data_dir.expanduser().resolve()
    data = load_analysis_inputs(base, arguments.mixture, arguments.run)
    data["mixture_index"] = arguments.mixture
    if arguments.results_dir is None:
        result_dir = base / "results" / (
            f"{data['mixture_id']}_run{arguments.run}_"
            "all_isotopes_clustered_first_representative"
        )
    else:
        result_dir = arguments.results_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Saved result directory not found: {result_dir}")

    posterior = load_saved_posterior(result_dir)
    validate_saved_analysis(data, posterior)
    context_dir = base / "Spectra_plots"
    context_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading posterior draws from: {result_dir}")
    print(
        f"Chains: {posterior['chains_A'].shape[0]}, retained draws per chain: "
        f"{posterior['chains_A'].shape[1]}"
    )

    if not arguments.skip_library_plots:
        activity_prior = posterior["metadata"]["model"]["priors"][
            "activity_inverse_gamma"
        ]
        plot_activity_prior(
            context_dir,
            alpha=float(activity_prior["alpha"]),
            beta_bq=float(activity_prior["beta_Bq"]),
            pi=float(
                posterior["metadata"]["model"]["priors"][
                    "isotope_inclusion_probability"
                ]
            ),
            show=arguments.show,
        )
        plot_original_template_panel(data, context_dir, show=arguments.show)
        plot_background_overlay(data, context_dir, show=arguments.show)

    plot_template_similarity(result_dir, show=arguments.show)
    reconstruction = reconstruct_posterior(data, posterior)
    plot_reconstruction(data, reconstruction, result_dir, show=arguments.show)
    plot_isotope_decomposition(
        data, posterior, reconstruction, result_dir, show=arguments.show
    )
    plot_traces_and_posteriors(
        posterior, reconstruction, result_dir, show=arguments.show
    )
    plot_template_overlays(
        data, posterior, reconstruction, result_dir, show=arguments.show
    )
    print(f"All figures saved to: {result_dir}")
    if not arguments.skip_library_plots:
        print(f"Context figures saved to: {context_dir}")


if __name__ == "__main__":
    main()
