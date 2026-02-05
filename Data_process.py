import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, expon, poisson
from scipy.special import gammaln
from scipy.special import gamma
import pymc as pm
import pytensor.tensor as at
import arviz as az
from scipy.optimize import nnls

base = Path("/Users/riana/Desktop/Research/Spectra_Bayes")

tmls = pd.read_excel(base / "02_A_REF.xlsx")
tref = pd.read_excel(base / "03_Templates_A_REF.xlsx")
ecal = pd.read_excel(base / "042_DATA_ECal.xlsx")
spectra = pd.read_excel(base / "04_DATA_spectra.xlsx")
all_spectra = spectra.iloc[:, 4:].to_numpy()
iso_names = np.array(['Na22','Mn54','Co57','Co60','Zn65','Y88','Cd109','Cs134','Cs137','Pb210','Am241'], dtype=str)
file_list = spectra['Filename'].astype(str).to_numpy()


# Build single spectra structure
single_spectra = [[] for _ in iso_names]
nuclide_list = tref['Nuclide'].astype(str).to_numpy()

for iso_index, iso in enumerate(iso_names):
    match = np.where(nuclide_list == iso)[0]
    msids = np.unique(tref['MeasurementSeriesID_613'].iloc[match].astype(str))


    for ms in msids:
        entry = {}
        entry['nm'] = iso
        entry['MSID'] = ms


        ind_msid = tref['MeasurementSeriesID_613'] == ms
        entry['A'] = tref.loc[ind_msid, 'A_Bq'].to_numpy()
        entry['uA'] = tref.loc[ind_msid, 'uA_Bq'].to_numpy()
        entry['Filename'] = tref.loc[ind_msid, 'Filename'].astype(str).to_numpy()


        counts_list = []
        lt_list = []


        for fname in entry['Filename']:
            idx = np.where(spectra['Filename'].astype(str).to_numpy() == fname)[0][0]
            counts_list.append(all_spectra[idx, :])
            lt_list.append(spectra['t_live_s'].iloc[idx])


        entry['Counts'] = np.vstack(counts_list)
        entry['Livetime'] = np.array(lt_list)


        ecal_list = []
        for fname in entry['Filename']:
            idx = np.where(ecal['Filename'].astype(str).to_numpy() == fname)[0][0]
            ecal_list.append(ecal.iloc[idx, 2:4].to_numpy())
        entry['Ecal'] = np.vstack(ecal_list)


        single_spectra[iso_index].append(entry)


# Background spectra
msid_list = spectra['MeasurementSeriesID_613'].astype(str).to_numpy()
is_background = np.array(['-NE-' in m for m in msid_list])
background_msids = np.unique(msid_list[is_background])


ne_spectra = []
for ms in background_msids:
    entry = {}
    entry['MSID'] = ms


    inds = np.where(msid_list == ms)[0]
    entry['Filename'] = spectra['Filename'].iloc[inds].astype(str).to_numpy()


    counts_list = []
    lt_list = []
    ecal_list = []


    for idx in inds:
        counts_list.append(all_spectra[idx, :])
        lt_list.append(spectra['t_live_s'].iloc[idx])


        idx_ec = np.where(ecal['Filename'].astype(str).to_numpy() == spectra['Filename'].iloc[idx])[0][0]
        ecal_list.append(ecal.iloc[idx_ec, 2:4].to_numpy())


    entry['Counts'] = np.vstack(counts_list)
    entry['Livetime'] = np.array(lt_list)
    entry['Ecal'] = np.vstack(ecal_list)


    ne_spectra.append(entry)


# Mixture spectra
mls_spectra = []
mls_ids = tmls['MeasurementSeriesID_613'].astype(str).unique()
mls_nuclide = tmls['Nuclide'].astype(str)


for ms in mls_ids:
    entry = {}
    entry['MSID'] = ms


    inds = np.where(tmls['MeasurementSeriesID_613'].astype(str).to_numpy() == ms)[0]
    filenames = np.unique(tmls['Filename'].iloc[inds].astype(str))
    entry['Filename'] = filenames


    counts = []
    lt = []
    ecal_vals = []


    for fname in filenames:
        idx = np.where(spectra['Filename'].astype(str).to_numpy() == fname)[0][0]
        counts.append(all_spectra[idx, :])
        lt.append(spectra['t_live_s'].iloc[idx])


        idx_ec = np.where(ecal['Filename'].astype(str).to_numpy() == fname)[0][0]
        ecal_vals.append(ecal.iloc[idx_ec, 2:4].to_numpy())


    entry['Counts'] = np.vstack(counts)
    entry['Livetime'] = np.array(lt)
    entry['Ecal'] = np.vstack(ecal_vals)


    mls_spectra.append(entry)


# Quick Sanity Check
print("Loaded:")
print(f"  {len(single_spectra)} isotopes")
print(f"  {len(ne_spectra)} background runs")
print(f"  {len(mls_spectra)} mixture spectra")


# Building templates for the isotopes X_all
def build_template_spectra_py(
    single_spectra,
    ne_spectra,
    channel_flag="channels",
):
    K = len(single_spectra)
    B = len(ne_spectra)
    if B == 0:
        raise ValueError("No background spectra found (ne_spectra is empty).")
    
    C = ne_spectra[0]["Counts"].shape[1]
    N_rep = [len(single_spectra[k]) for k in range(K)]
    M = sum(N_rep)

    X_iso = np.zeros((C, M), dtype=float)  # cps/Bq
    X_bg  = np.zeros((C, B), dtype=float)  # cps

    iso_of_col = np.zeros(M, dtype=int)
    iso_template_id = []
    activities_meta = np.zeros(M, dtype=float) 
    uactivities_rel_meta = np.zeros(M, dtype=float)

    bg_template_id = []
    iso_index = [[] for _ in range(K)]

    col = 0
    for j in range(K):
        for n in range(N_rep[j]):
            entry = single_spectra[j][n]
            Cnts = entry["Counts"]
            LT   = entry["Livetime"]
            A    = entry["A"]
            uA   = entry.get("uA", None)

            valid = (
                np.isfinite(LT) & (LT > 0) &
                np.isfinite(A)  & (A > 0)
            )
            if valid.sum() == 0:
                raise ValueError(
                    f"No valid runs with LT>0 and A>0 for isotope j={j}, replicate n={n}, MSID={entry.get('MSID','NA')}"
                )

            Cnts_v = Cnts[valid, :]
            LT_v   = LT[valid]
            A_v    = A[valid]

            rate = Cnts_v / LT_v[:, None]
            rate_per_bq = rate / A_v[:, None]
            w = LT_v / LT_v.sum()
            template = (w[:, None] * rate_per_bq).sum(axis=0)
            X_iso[:, col] = template

            iso_of_col[col] = j
            iso_index[j].append(col)
            iso_template_id.append(entry.get("MSID", f"iso{j}_rep{n}"))
            activities_meta[col] = np.median(A_v)
            if uA is not None:
                uA_v = np.asarray(uA)[valid]
                uactivities_rel_meta[col] = np.median(uA_v / A_v)
            else:
                uactivities_rel_meta[col] = np.nan

            col += 1

    assert col == M

    for b, bg_entry in enumerate(ne_spectra):
        C_bg  = bg_entry["Counts"]
        LT_bg = bg_entry["Livetime"]

        valid_bg = np.isfinite(LT_bg) & (LT_bg > 0)
        if valid_bg.sum() == 0:
            raise ValueError(f"No valid background runs with LT>0 for background b={b}, MSID={bg_entry.get('MSID','NA')}")

        C_bg_v  = C_bg[valid_bg, :]
        LT_bg_v = LT_bg[valid_bg]

        bg_rate = C_bg_v.sum(axis=0) / LT_bg_v.sum()
        X_bg[:, b] = bg_rate

        bg_template_id.append(f"Background_{bg_entry.get('MSID', b)}")
    ee = np.arange(1, C + 1)
    F_iso = [PchipInterpolator(ee, X_iso[:, m]) for m in range(M)]
    F_bg  = [PchipInterpolator(ee, X_bg[:, b])  for b in range(B)]

    info = {
        "channel_flag": channel_flag,
        "C": C,
        "K": K,
        "B": B,
        "M": M,
        "iso_of_col": iso_of_col,
        "iso_index": iso_index,
        "iso_template_id": iso_template_id,
        "bg_template_id": bg_template_id,
        "activities_meta": activities_meta,
        "uactivities_rel_meta": uactivities_rel_meta,
        "F_iso": F_iso,
        "F_bg": F_bg,
    }

    return X_iso, X_bg, info

X_iso, X_bg, info = build_template_spectra_py(
    single_spectra=single_spectra,
    ne_spectra=ne_spectra,
    channel_flag="channels"
)

# Quick sanity check
print("X_iso shape (C, M):", X_iso.shape)
print("X_bg shape  (C, B):", X_bg.shape)
print("Total replicate templates M:", info["M"])
print("Replicate columns for isotope 0:", info["iso_index"][0])

C, M = X_iso.shape
K = len(info["iso_index"])
X_iso_avg = np.zeros((C, K), dtype=float)
for j in range(K):
    cols_j = info["iso_index"][j]
    X_iso_avg[:, j] = X_iso[:, cols_j].mean(axis=1)

print("X_iso_avg shape (C, K):", X_iso_avg.shape)


C = X_iso_avg.shape[0]
K = X_iso_avg.shape[1]

channels = np.arange(1, C + 1)

F_iso_avg = [PchipInterpolator(channels, X_iso_avg[:, j]) for j in range(K)]
F_bg      = info["F_bg"]
B = len(F_bg)

