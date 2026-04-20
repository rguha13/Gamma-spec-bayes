"""Template construction for Bayesian full-spectrum analysis."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class TemplateLibrary:
    """Normalized isotope and background templates."""

    isotope_templates: np.ndarray
    background_templates: np.ndarray
    averaged_isotope_templates: np.ndarray
    isotope_interpolators: list
    background_interpolators: list
    channels: np.ndarray
    iso_index: list
    iso_of_column: np.ndarray
    isotope_template_ids: list


def build_template_matrices(single_spectra: list, background_spectra: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build isotope templates in cps/Bq and background templates in cps."""

    k_isotopes = len(single_spectra)
    n_background = len(background_spectra)
    if n_background == 0:
        raise ValueError("No background spectra found.")

    n_channels = background_spectra[0]["Counts"].shape[1]
    n_replicates = [len(single_spectra[k]) for k in range(k_isotopes)]
    n_isotope_templates = sum(n_replicates)

    x_iso = np.zeros((n_channels, n_isotope_templates), dtype=float)
    x_bg = np.zeros((n_channels, n_background), dtype=float)
    iso_of_col = np.zeros(n_isotope_templates, dtype=int)
    iso_template_ids = []
    iso_index = [[] for _ in range(k_isotopes)]

    col = 0
    for j in range(k_isotopes):
        for n in range(n_replicates[j]):
            entry = single_spectra[j][n]
            counts = entry["Counts"]
            livetimes = entry["Livetime"]
            activities = entry["A"]

            valid = np.isfinite(livetimes) & (livetimes > 0) & np.isfinite(activities) & (activities > 0)
            if valid.sum() == 0:
                raise ValueError(f"No valid calibration runs for isotope index {j}, replicate {n}.")

            counts_valid = counts[valid, :]
            livetimes_valid = livetimes[valid]
            activities_valid = activities[valid]

            # Convert counts to count rate per becquerel, then livetime-average
            # repeated calibration runs for the same isotope template.
            rate_per_bq = (counts_valid / livetimes_valid[:, None]) / activities_valid[:, None]
            weights = livetimes_valid / livetimes_valid.sum()
            x_iso[:, col] = (weights[:, None] * rate_per_bq).sum(axis=0)

            iso_of_col[col] = j
            iso_index[j].append(col)
            iso_template_ids.append(entry.get("MSID", f"iso{j}_rep{n}"))
            col += 1

    for b, bg_entry in enumerate(background_spectra):
        counts_bg = bg_entry["Counts"]
        livetimes_bg = bg_entry["Livetime"]
        valid_bg = np.isfinite(livetimes_bg) & (livetimes_bg > 0)
        if valid_bg.sum() == 0:
            raise ValueError(f"No valid background runs for background index {b}.")

        # Background templates are empirical count-rate spectra.
        x_bg[:, b] = counts_bg[valid_bg, :].sum(axis=0) / livetimes_bg[valid_bg].sum()

    info = {
        "iso_index": iso_index,
        "iso_of_col": iso_of_col,
        "isotope_template_ids": iso_template_ids,
    }
    return x_iso, x_bg, info


def build_template_library(single_spectra: list, background_spectra: list) -> TemplateLibrary:
    """Build averaged isotope templates and PCHIP interpolators."""

    x_iso, x_bg, info = build_template_matrices(single_spectra, background_spectra)
    n_channels, _ = x_iso.shape
    k_isotopes = len(info["iso_index"])
    channels = np.arange(1, n_channels + 1)

    x_iso_avg = np.zeros((n_channels, k_isotopes), dtype=float)
    for j, cols_j in enumerate(info["iso_index"]):
        x_iso_avg[:, j] = x_iso[:, cols_j].mean(axis=1)

    isotope_interpolators = [
        PchipInterpolator(channels, x_iso_avg[:, j]) for j in range(k_isotopes)
    ]
    background_interpolators = [
        PchipInterpolator(channels, x_bg[:, b]) for b in range(x_bg.shape[1])
    ]

    return TemplateLibrary(
        isotope_templates=x_iso,
        background_templates=x_bg,
        averaged_isotope_templates=x_iso_avg,
        isotope_interpolators=isotope_interpolators,
        background_interpolators=background_interpolators,
        channels=channels,
        iso_index=info["iso_index"],
        iso_of_column=info["iso_of_col"],
        isotope_template_ids=info["isotope_template_ids"],
    )

