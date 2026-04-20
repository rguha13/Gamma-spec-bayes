"""Data loading and reshaping utilities for the Wubbeler gamma dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """Container holding raw input tables and spectrum groupings."""

    activity_refs: pd.DataFrame
    template_refs: pd.DataFrame
    energy_calibration: pd.DataFrame
    spectra: pd.DataFrame
    half_lives: pd.DataFrame
    single_spectra: list
    background_spectra: list
    mixture_spectra: list


def load_input_tables(data_dir: Path) -> tuple[pd.DataFrame, ...]:
    """Read the required dataset tables from ``data_dir``."""

    data_dir = Path(data_dir)
    activity_refs = pd.read_excel(data_dir / "02_A_REF.xlsx")
    template_refs = pd.read_excel(data_dir / "03_Templates_A_REF.xlsx")
    energy_calibration = pd.read_excel(data_dir / "042_DATA_ECal.xlsx")
    spectra = pd.read_excel(data_dir / "04_DATA_spectra.xlsx")
    half_lives = pd.read_csv(data_dir / "05_Half_lives.csv")
    return activity_refs, template_refs, energy_calibration, spectra, half_lives


def _lookup_row_index(values: np.ndarray, value: str, label: str) -> int:
    matches = np.where(values == value)[0]
    if matches.size != 1:
        raise ValueError(f"Expected exactly one {label} row for {value}; found {matches.size}.")
    return int(matches[0])


def build_single_spectra(
    template_refs: pd.DataFrame,
    spectra: pd.DataFrame,
    energy_calibration: pd.DataFrame,
    isotope_names: np.ndarray,
) -> list:
    """Group single-isotope calibration runs by isotope and measurement series."""

    all_counts = spectra.iloc[:, 4:].to_numpy()
    spectrum_files = spectra["Filename"].astype(str).to_numpy()
    ecal_files = energy_calibration["Filename"].astype(str).to_numpy()
    nuclide_list = template_refs["Nuclide"].astype(str).to_numpy()

    single_spectra = [[] for _ in isotope_names]
    for iso_index, isotope in enumerate(isotope_names):
        match = np.where(nuclide_list == isotope)[0]
        msids = np.unique(template_refs["MeasurementSeriesID_613"].iloc[match].astype(str))

        for msid in msids:
            entry = {"nm": isotope, "MSID": msid}
            ind_msid = template_refs["MeasurementSeriesID_613"] == msid
            entry["A"] = template_refs.loc[ind_msid, "A_Bq"].to_numpy()
            entry["uA"] = template_refs.loc[ind_msid, "uA_Bq"].to_numpy()
            entry["Filename"] = template_refs.loc[ind_msid, "Filename"].astype(str).to_numpy()

            counts_list, livetime_list, ecal_list = [], [], []
            for filename in entry["Filename"]:
                spectrum_idx = _lookup_row_index(spectrum_files, filename, "spectrum")
                ecal_idx = _lookup_row_index(ecal_files, filename, "energy-calibration")
                counts_list.append(all_counts[spectrum_idx, :])
                livetime_list.append(spectra["t_live_s"].iloc[spectrum_idx])
                ecal_list.append(energy_calibration.iloc[ecal_idx, 2:4].to_numpy())

            entry["Counts"] = np.vstack(counts_list)
            entry["Livetime"] = np.asarray(livetime_list, dtype=float)
            entry["Ecal"] = np.vstack(ecal_list)
            single_spectra[iso_index].append(entry)

    return single_spectra


def build_background_spectra(
    spectra: pd.DataFrame,
    energy_calibration: pd.DataFrame,
) -> list:
    """Collect empirical background-only spectra."""

    all_counts = spectra.iloc[:, 4:].to_numpy()
    msid_list = spectra["MeasurementSeriesID_613"].astype(str).to_numpy()
    ecal_files = energy_calibration["Filename"].astype(str).to_numpy()
    is_background = np.array(["-NE-" in msid for msid in msid_list])
    background_msids = np.unique(msid_list[is_background])

    background_spectra = []
    for msid in background_msids:
        entry = {"MSID": msid}
        inds = np.where(msid_list == msid)[0]
        entry["Filename"] = spectra["Filename"].iloc[inds].astype(str).to_numpy()

        counts_list, livetime_list, ecal_list = [], [], []
        for idx in inds:
            filename = str(spectra["Filename"].iloc[idx])
            ecal_idx = _lookup_row_index(ecal_files, filename, "energy-calibration")
            counts_list.append(all_counts[idx, :])
            livetime_list.append(spectra["t_live_s"].iloc[idx])
            ecal_list.append(energy_calibration.iloc[ecal_idx, 2:4].to_numpy())

        entry["Counts"] = np.vstack(counts_list)
        entry["Livetime"] = np.asarray(livetime_list, dtype=float)
        entry["Ecal"] = np.vstack(ecal_list)
        background_spectra.append(entry)

    return background_spectra


def build_mixture_spectra(
    activity_refs: pd.DataFrame,
    spectra: pd.DataFrame,
    energy_calibration: pd.DataFrame,
) -> list:
    """Collect mixture spectra by measurement series."""

    all_counts = spectra.iloc[:, 4:].to_numpy()
    spectrum_files = spectra["Filename"].astype(str).to_numpy()
    ecal_files = energy_calibration["Filename"].astype(str).to_numpy()
    mixture_spectra = []

    for msid in activity_refs["MeasurementSeriesID_613"].astype(str).unique():
        entry = {"MSID": msid}
        inds = np.where(activity_refs["MeasurementSeriesID_613"].astype(str).to_numpy() == msid)[0]
        filenames = np.unique(activity_refs["Filename"].iloc[inds].astype(str))
        entry["Filename"] = filenames

        counts_list, livetime_list, ecal_list = [], [], []
        for filename in filenames:
            spectrum_idx = _lookup_row_index(spectrum_files, filename, "spectrum")
            ecal_idx = _lookup_row_index(ecal_files, filename, "energy-calibration")
            counts_list.append(all_counts[spectrum_idx, :])
            livetime_list.append(spectra["t_live_s"].iloc[spectrum_idx])
            ecal_list.append(energy_calibration.iloc[ecal_idx, 2:4].to_numpy())

        entry["Counts"] = np.vstack(counts_list)
        entry["Livetime"] = np.asarray(livetime_list, dtype=float)
        entry["Ecal"] = np.vstack(ecal_list)
        mixture_spectra.append(entry)

    return mixture_spectra


def load_dataset(data_dir: Path, isotope_names: np.ndarray) -> DataBundle:
    """Load all input tables and construct spectrum groupings."""

    activity_refs, template_refs, energy_calibration, spectra, half_lives = load_input_tables(data_dir)
    single_spectra = build_single_spectra(template_refs, spectra, energy_calibration, isotope_names)
    background_spectra = build_background_spectra(spectra, energy_calibration)
    mixture_spectra = build_mixture_spectra(activity_refs, spectra, energy_calibration)

    return DataBundle(
        activity_refs=activity_refs,
        template_refs=template_refs,
        energy_calibration=energy_calibration,
        spectra=spectra,
        half_lives=half_lives,
        single_spectra=single_spectra,
        background_spectra=background_spectra,
        mixture_spectra=mixture_spectra,
    )

