# =============================================================================
# Bayesian Full-Spectrum Analysis for Radioisotope Identification
# =============================================================================
# IEEE Transactions on Nuclear Science — Riana Guha et al.
#
# Final clustered-template model:
#   All 11 isotopes and all 31 original templates enter the candidate library.
#   Near-duplicate templates are clustered by cosine similarity and Pearson
#   correlation. The first member represents each cluster, and cluster
#   contributions enter through simplex weights.
#   Activities, inclusion indicators, and shifts are sampled at isotope level.
#   Isotope/background energy shifts use theta ~ N(0, 0.5^2) keV.
#   R-hat and ESS diagnostics are calculated with ArviZ.
#   Activities are compared with PeakAnalysis_G8 references from file 021.
# =============================================================================

import argparse
import os
import time
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaln
from pathlib import Path
import arviz as az

# =============================================================================
# PATHS & DATA LOADING
# =============================================================================


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Bayesian full-spectrum analysis of one HPGe mixture spectrum."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=project_dir,
        help="Directory containing the Wübbeler et al. data files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root (default: <data-dir>/results).",
    )
    parser.add_argument("--mixture-index", type=int, default=1)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=5000)
    return parser.parse_args()


args = parse_args()
base = args.data_dir.expanduser().resolve()

save_root = (
    args.output_dir.expanduser().resolve()
    if args.output_dir is not None else base / "results"
)
os.makedirs(save_root, exist_ok=True)

required_data_files = [
    "02_A_REF.xlsx", "03_Templates_A_REF.xlsx", "042_DATA_ECal.xlsx",
    "04_DATA_spectra.xlsx", "021_A_REF_main.csv", "05_Half_lives.csv",
]
missing_data_files = [name for name in required_data_files if not (base / name).is_file()]
if missing_data_files:
    raise FileNotFoundError(
        f"Missing required data files in {base}: {missing_data_files}"
    )

tmls      = pd.read_excel(base / "02_A_REF.xlsx")
tref      = pd.read_excel(base / "03_Templates_A_REF.xlsx")
ecal      = pd.read_excel(base / "042_DATA_ECal.xlsx")
spectra   = pd.read_excel(base / "04_DATA_spectra.xlsx")

all_spectra = spectra.iloc[:, 4:].to_numpy()
spectra_files = spectra['Filename'].astype(str).to_numpy()
ecal_files = ecal['Filename'].astype(str).to_numpy()
ecal_columns = ["Ecal_b0_GX", "Ecal_b1_GX"]
if pd.Series(ecal_files).duplicated().any():
    raise ValueError("042_DATA_ECal.xlsx contains duplicate filenames.")
iso_names   = np.array(
    ['Na22','Mn54','Co57','Co60','Zn65','Y88','Cd109','Cs134','Cs137','Pb210','Am241'],
    dtype=str
)
nuclide_list = tref['Nuclide'].astype(str).to_numpy()

# =============================================================================
# BUILD SINGLE-ISOTOPE SPECTRA STRUCTURES
# =============================================================================
single_spectra = [[] for _ in iso_names]

for iso_index, iso in enumerate(iso_names):
    match   = np.where(nuclide_list == iso)[0]
    msids   = np.unique(tref['MeasurementSeriesID_613'].iloc[match].astype(str))

    for ms in msids:
        entry = {}
        entry['MSID'] = ms

        ind_msid        = tref['MeasurementSeriesID_613'] == ms
        entry['A']      = tref.loc[ind_msid, 'A_Bq'].to_numpy()
        entry['Filename'] = tref.loc[ind_msid, 'Filename'].astype(str).to_numpy()

        counts_list, lt_list, ecal_list = [], [], []
        for fname in entry['Filename']:
            idx = np.where(spectra_files == fname)[0][0]
            idx_ec = np.where(ecal_files == fname)[0][0]
            counts_list.append(all_spectra[idx, :])
            lt_list.append(spectra['t_live_s'].iloc[idx])
            ecal_list.append(
                ecal.loc[idx_ec, ecal_columns].to_numpy(dtype=float)
            )

        entry['Counts']   = np.vstack(counts_list)
        entry['Livetime'] = np.array(lt_list)
        entry['Ecal']     = np.vstack(ecal_list)
        single_spectra[iso_index].append(entry)

# Background spectra
msid_list      = spectra['MeasurementSeriesID_613'].astype(str).to_numpy()
is_background  = np.array(['-NE-' in m for m in msid_list])
background_msids = np.unique(msid_list[is_background])

ne_spectra = []
for ms in background_msids:
    entry = {}
    entry['MSID'] = ms
    inds          = np.where(msid_list == ms)[0]
    entry['Filename'] = spectra['Filename'].iloc[inds].astype(str).to_numpy()

    counts_list, lt_list, ecal_list = [], [], []
    for idx in inds:
        fname = str(spectra['Filename'].iloc[idx])
        idx_ec = np.where(ecal_files == fname)[0][0]
        counts_list.append(all_spectra[idx, :])
        lt_list.append(spectra['t_live_s'].iloc[idx])
        ecal_list.append(
            ecal.loc[idx_ec, ecal_columns].to_numpy(dtype=float)
        )

    entry['Counts']   = np.vstack(counts_list)
    entry['Livetime'] = np.array(lt_list)
    entry['Ecal']     = np.vstack(ecal_list)
    ne_spectra.append(entry)

# Mixture spectra
mls_spectra = []
mls_ids     = tmls['MeasurementSeriesID_613'].astype(str).unique()

for ms in mls_ids:
    entry = {}
    entry['MSID'] = ms
    inds          = np.where(tmls['MeasurementSeriesID_613'].astype(str).to_numpy() == ms)[0]
    filenames     = np.unique(tmls['Filename'].iloc[inds].astype(str))
    entry['Filename'] = filenames

    counts, lt, ecal_vals = [], [], []
    for fname in filenames:
        idx = np.where(spectra_files == fname)[0][0]
        counts.append(all_spectra[idx, :])
        lt.append(spectra['t_live_s'].iloc[idx])
        idx_ec = np.where(ecal_files == fname)[0][0]
        ecal_vals.append(
            ecal.loc[idx_ec, ecal_columns].to_numpy(dtype=float)
        )

    entry['Counts']   = np.vstack(counts)
    entry['Livetime'] = np.array(lt)
    entry['Ecal']     = np.vstack(ecal_vals)
    mls_spectra.append(entry)

print("Loaded:")
print(f"  {len(single_spectra)} isotopes")
print(f"  {len(ne_spectra)} background runs")
print(f"  {len(mls_spectra)} mixture spectra")

# =============================================================================
# SELECT MIXTURE AND DEFINE ITS RUN-SPECIFIC ENERGY GRID
# =============================================================================
i_mix = args.mixture_index
r_run = args.run_index

mix0 = mls_spectra[i_mix]
Y0   = mix0["Counts"][r_run, :].astype(float)
T0   = float(mix0["Livetime"][r_run])

C = Y0.size
channels = np.arange(1, C + 1, dtype=float)
ecal_run0 = mix0["Ecal"][r_run, :]
a_cal, b_cal = float(ecal_run0[0]), float(ecal_run0[1])
energy_template = a_cal + b_cal * channels
energy = energy_template.copy()

save_dir = save_root / f"{mix0['MSID']}_run{r_run}_all_isotopes_clustered_first_representative"
os.makedirs(save_dir, exist_ok=True)

# =============================================================================
# BUILD TEMPLATES
# =============================================================================
def register_to_energy_grid(values, calibration, channels, common_energy):
    """Register one native-channel spectrum on the selected run's grid."""
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


def build_template_spectra_py(single_spectra, ne_spectra, channels,
                              common_energy):
    K = len(single_spectra)
    B = len(ne_spectra)
    if B == 0:
        raise ValueError("No background spectra found.")

    C     = ne_spectra[0]["Counts"].shape[1]
    N_rep = [len(single_spectra[k]) for k in range(K)]
    M     = sum(N_rep)

    X_iso = np.zeros((C, M), dtype=float)
    X_bg  = np.zeros((C, B), dtype=float)

    iso_of_col         = np.zeros(M, dtype=int)
    iso_template_id    = []
    iso_index          = [[] for _ in range(K)]
    template_livetime  = np.zeros(M, dtype=float)
    background_livetime = np.zeros(B, dtype=float)

    col = 0
    for j in range(K):
        for n in range(N_rep[j]):
            entry = single_spectra[j][n]
            Cnts  = entry["Counts"]
            LT    = entry["Livetime"]
            A     = entry["A"]
            Ecal  = entry["Ecal"]

            valid = (
                np.isfinite(LT) & (LT > 0) & np.isfinite(A) & (A > 0)
                & np.all(np.isfinite(Ecal), axis=1) & (Ecal[:, 1] > 0)
            )
            if valid.sum() == 0:
                raise ValueError(f"No valid runs for isotope j={j}, replicate n={n}.")

            Cnts_v       = Cnts[valid, :]
            LT_v         = LT[valid]
            A_v          = A[valid]
            Ecal_v       = Ecal[valid, :]
            rate_per_bq  = (Cnts_v / LT_v[:, None]) / A_v[:, None]
            registered_runs = np.vstack([
                register_to_energy_grid(
                    rate_per_bq[r], Ecal_v[r], channels, common_energy
                )
                for r in range(len(LT_v))
            ])
            w            = LT_v / LT_v.sum()
            X_iso[:, col] = (w[:, None] * registered_runs).sum(axis=0)
            template_livetime[col] = LT_v.sum()

            iso_of_col[col]  = j
            iso_index[j].append(col)
            iso_template_id.append(entry.get("MSID", f"iso{j}_rep{n}"))
            col += 1

    assert col == M

    for b, bg_entry in enumerate(ne_spectra):
        C_bg    = bg_entry["Counts"]
        LT_bg   = bg_entry["Livetime"]
        Ecal_bg = bg_entry["Ecal"]
        valid_bg = (
            np.isfinite(LT_bg) & (LT_bg > 0)
            & np.all(np.isfinite(Ecal_bg), axis=1) & (Ecal_bg[:, 1] > 0)
        )
        if valid_bg.sum() == 0:
            raise ValueError(f"No valid background runs for b={b}.")
        C_bg_v = C_bg[valid_bg, :]
        LT_bg_v = LT_bg[valid_bg]
        Ecal_bg_v = Ecal_bg[valid_bg, :]
        registered_bg_runs = np.vstack([
            register_to_energy_grid(
                C_bg_v[r] / LT_bg_v[r], Ecal_bg_v[r], channels,
                common_energy
            )
            for r in range(len(LT_bg_v))
        ])
        w_bg = LT_bg_v / LT_bg_v.sum()
        X_bg[:, b] = (w_bg[:, None] * registered_bg_runs).sum(axis=0)
        background_livetime[b] = LT_bg_v.sum()

    info = {
        "iso_of_col": iso_of_col, "iso_index": iso_index,
        "iso_template_id": iso_template_id,
        "template_livetime": template_livetime,
        "background_livetime": background_livetime,
    }
    return X_iso, X_bg, info


X_library, X_bg, info = build_template_spectra_py(
    single_spectra, ne_spectra, channels, energy_template
)
print(f"Registered full template library shape (C, M): {X_library.shape}")
print(f"X_bg  shape (C, B): {X_bg.shape}")

C, M_library = X_library.shape
library_iso_of_template = info["iso_of_col"]
library_template_labels = np.array([
    f"{iso_names[library_iso_of_template[m]]}_{info['iso_template_id'][m]}"
    for m in range(M_library)
], dtype=str)

# Use all 11 isotope families and all 31 original templates as clustering
# candidates. Only the first member of each within-isotope cluster enters X.
model_isotope_names = iso_names.copy()
model_iso_global_idx = np.arange(len(iso_names), dtype=int)
model_template_global_idx = np.concatenate([
    np.asarray(info["iso_index"][global_j], dtype=int)
    for global_j in model_iso_global_idx
])

X_candidate = X_library[:, model_template_global_idx]
candidate_template_labels = library_template_labels[model_template_global_idx]
candidate_template_livetime = info["template_livetime"][
    model_template_global_idx
]
iso_of_candidate_global = library_iso_of_template[model_template_global_idx]
iso_of_candidate_model = np.array([
    int(np.where(model_iso_global_idx == global_j)[0][0])
    for global_j in iso_of_candidate_global
], dtype=int)

K = len(model_isotope_names)

# =============================================================================
# COSINE-SIMILARITY MAP AND WITHIN-ISOTOPE CLUSTERING
# =============================================================================
similarity_energy_min = 20.0
similarity_energy_max = 1500.0
cosine_cluster_threshold = 0.999
pearson_cluster_threshold = 0.999
similarity_mask = (
    (energy_template >= similarity_energy_min)
    & (energy_template <= similarity_energy_max)
)

candidate_vectors = X_candidate[similarity_mask, :].T
candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
if np.any(~np.isfinite(candidate_norms)) or np.any(candidate_norms <= 0):
    raise ValueError("Every candidate template must have a finite nonzero norm.")
candidate_unit_vectors = candidate_vectors / candidate_norms[:, None]
cosine_similarity = np.clip(
    candidate_unit_vectors @ candidate_unit_vectors.T, -1.0, 1.0
)
pearson_correlation = np.corrcoef(candidate_vectors)

similarity_df = pd.DataFrame(
    cosine_similarity,
    index=candidate_template_labels,
    columns=candidate_template_labels,
)
similarity_df.to_csv(save_dir / "template_cosine_similarity.csv")
pd.DataFrame(
    pearson_correlation,
    index=candidate_template_labels,
    columns=candidate_template_labels,
).to_csv(save_dir / "template_pearson_correlation.csv")

# Deterministic representative-based clustering. Candidate templates are
# processed in their existing order. A template joins the first cluster for
# its isotope whose representative has similarity at or above the threshold;
# otherwise it starts a new cluster. The representative is therefore always
# the first member of a cluster, as requested.
template_clusters = []
candidate_cluster_id = np.full(len(candidate_template_labels), -1, dtype=int)
for j in range(K):
    isotope_candidates = np.where(iso_of_candidate_model == j)[0]
    isotope_cluster_ids = []
    for candidate_m in isotope_candidates:
        assigned_cluster = None
        for cluster_id in isotope_cluster_ids:
            representative_m = template_clusters[cluster_id][0]
            if (
                cosine_similarity[candidate_m, representative_m]
                >= cosine_cluster_threshold
                and pearson_correlation[candidate_m, representative_m]
                >= pearson_cluster_threshold
            ):
                assigned_cluster = cluster_id
                break
        if assigned_cluster is None:
            assigned_cluster = len(template_clusters)
            template_clusters.append([int(candidate_m)])
            isotope_cluster_ids.append(assigned_cluster)
        else:
            template_clusters[assigned_cluster].append(int(candidate_m))
        candidate_cluster_id[candidate_m] = assigned_cluster

cluster_rows = []
for cluster_id, members in enumerate(template_clusters):
    representative_m = members[0]
    member_livetimes = candidate_template_livetime[members]
    normalized_livetime_weights = member_livetimes / member_livetimes.sum()
    for local_id, candidate_m in enumerate(members):
        cluster_rows.append({
            "Cluster": cluster_id,
            "Isotope": model_isotope_names[
                iso_of_candidate_model[candidate_m]
            ],
            "Candidate_index": candidate_m,
            "Library_column": int(model_template_global_idx[candidate_m]),
            "Template": candidate_template_labels[candidate_m],
            "Livetime_s": candidate_template_livetime[candidate_m],
            "Cluster_livetime_weight": normalized_livetime_weights[local_id],
            "Representative": candidate_m == representative_m,
            "Representative_template": candidate_template_labels[
                representative_m
            ],
            "Cosine_to_representative": cosine_similarity[
                candidate_m, representative_m
            ],
            "Pearson_to_representative": pearson_correlation[
                candidate_m, representative_m
            ],
            "Cosine_threshold": cosine_cluster_threshold,
            "Pearson_threshold": pearson_cluster_threshold,
            "Energy_min_keV": similarity_energy_min,
            "Energy_max_keV": similarity_energy_max,
        })
cluster_table = pd.DataFrame(cluster_rows)
cluster_table.to_csv(save_dir / "template_clusters.csv", index=False)

cluster_members = [np.asarray(members, dtype=int) for members in template_clusters]
representative_candidate_idx = np.array(
    [members[0] for members in cluster_members], dtype=int
)
X_template = X_candidate[:, representative_candidate_idx]
iso_of_template_model = np.array([
    iso_of_candidate_model[members[0]] for members in cluster_members
], dtype=int)
template_labels = candidate_template_labels[representative_candidate_idx]
M = X_template.shape[1]
model_iso_index = [
    np.where(iso_of_template_model == j)[0].astype(int)
    for j in range(K)
]

# Cluster the measured backgrounds by the same pairwise shape checks and keep
# the first measurement in each cluster. No background averaging is applied.
B_library = X_bg.shape[1]
background_cosine_threshold = 0.995
background_pearson_threshold = 0.995
background_vectors = X_bg[similarity_mask, :].T
background_norms = np.linalg.norm(background_vectors, axis=1)
background_cosine_similarity = np.clip(
    (background_vectors / background_norms[:, None])
    @ (background_vectors / background_norms[:, None]).T,
    -1.0, 1.0
)
background_pearson_correlation = (
    np.ones((1, 1), dtype=float) if B_library == 1
    else np.corrcoef(background_vectors)
)
pd.DataFrame(
    background_cosine_similarity,
    index=[f"Background_{b}" for b in range(B_library)],
    columns=[f"Background_{b}" for b in range(B_library)],
).to_csv(save_dir / "background_cosine_similarity.csv")
pd.DataFrame(
    background_pearson_correlation,
    index=[f"Background_{b}" for b in range(B_library)],
    columns=[f"Background_{b}" for b in range(B_library)],
).to_csv(save_dir / "background_pearson_correlation.csv")

background_clusters = []
background_cluster_id = np.full(B_library, -1, dtype=int)
for background_id in range(B_library):
    assigned_cluster = None
    for cluster_id, members in enumerate(background_clusters):
        representative_id = members[0]
        if (
            background_cosine_similarity[background_id, representative_id]
            >= background_cosine_threshold
            and background_pearson_correlation[background_id, representative_id]
            >= background_pearson_threshold
        ):
            assigned_cluster = cluster_id
            break
    if assigned_cluster is None:
        assigned_cluster = len(background_clusters)
        background_clusters.append([background_id])
    else:
        background_clusters[assigned_cluster].append(background_id)
    background_cluster_id[background_id] = assigned_cluster

background_representative_idx = np.array(
    [members[0] for members in background_clusters], dtype=int
)
X_background = X_bg[:, background_representative_idx]
background_labels = np.array([
    f"Background_{background_id}"
    for background_id in background_representative_idx
], dtype=str)

background_cluster_rows = []
for cluster_id, members in enumerate(background_clusters):
    representative_id = members[0]
    for background_id in members:
        background_cluster_rows.append({
            "Cluster": cluster_id,
            "Background_index": background_id,
            "Background": f"Background_{background_id}",
            "Representative": background_id == representative_id,
            "Representative_background": f"Background_{representative_id}",
            "Cosine_to_representative": background_cosine_similarity[
                background_id, representative_id
            ],
            "Pearson_to_representative": background_pearson_correlation[
                background_id, representative_id
            ],
            "Cosine_threshold": background_cosine_threshold,
            "Pearson_threshold": background_pearson_threshold,
        })
pd.DataFrame(background_cluster_rows).to_csv(
    save_dir / "background_clusters.csv", index=False
)

B = X_background.shape[1]
P          = K + B
background_shift_labels = background_labels.copy()
alignment_labels = np.concatenate([
    model_isotope_names, background_shift_labels
])

# =============================================================================
# INTERPOLATORS AND HYPERPARAMETERS
# =============================================================================
F_template_energy = [
    PchipInterpolator(energy_template, X_template[:, m], extrapolate=True)
    for m in range(M)
]
F_bg_energy = [
    PchipInterpolator(energy_template, X_background[:, b], extrapolate=True)
    for b in range(B)
]

# Prior hyperparameters
alpha_ig    = 2.5
beta_ig     = 4500      # Bq; IG prior on activity (mean = 3000 Bq)
pi_iso_prior = 0.5
pi = pi_iso_prior * np.ones(K)
sigma_theta = 0.5  # keV; isotope-level and background shifts
s_gamma     = 0.5 * np.ones(B)
weight_dirichlet_alpha = 1.0
weight_proposal_concentration = 2000.0 * np.ones(K)

# Isotope-specific Dirichlet proposal concentrations. A smaller concentration
# produces larger simplex moves; a larger concentration produces smaller moves.
weight_proposal_concentration[model_isotope_names == "Co57"] = 750.0
weight_proposal_concentration[model_isotope_names == "Cd109"] = 5000.0

# Proposal scales
step_logA = 0.01 * np.ones(K)
step_logA[model_isotope_names == "Cd109"] = 0.003

step_gamma = 0.002 * np.ones(B)

step_theta = 0.006 * np.ones(K + B)
step_theta[np.where(model_isotope_names == "Cd109")[0][0]] = 0.0015
step_theta[np.where(model_isotope_names == "Pb210")[0][0]] = 0.005
step_theta[K:] = 0.001

print(f"\nSelected mixture : {mix0['MSID']}")
print(f"Mixture index     : {i_mix} (0-indexed)")
print(f"Selected run      : {r_run} ({mix0['Filename'][r_run]})")
print(f"042 calibration   : b0 = {a_cal:.9g} keV, b1 = {b_cal:.9g} keV/channel")
print(f"Candidate isotopes : {', '.join(model_isotope_names)}")
print(
    f"Template clustering: cosine>={cosine_cluster_threshold:.6f}, "
    f"Pearson>={pearson_cluster_threshold:.6f}, "
    f"energy={similarity_energy_min:g}-{similarity_energy_max:g} keV"
)
print(
    f"Template reduction : {X_candidate.shape[1]} candidates -> "
    f"{M} shape clusters"
)
for j, isotope in enumerate(model_isotope_names):
    candidate_count = int((iso_of_candidate_model == j).sum())
    print(
        f"  {isotope:6s}: {candidate_count} candidates -> "
        f"{len(model_iso_index[j])} clusters"
    )
for cluster_id, members in enumerate(template_clusters):
    labels = ", ".join(candidate_template_labels[m] for m in members)
    print(
        f"  cluster {cluster_id:02d}: representative="
        f"{candidate_template_labels[members[0]]}; members={labels}"
    )
print(f"B = {B} background components")
print(
    f"Background clustering: cosine>={background_cosine_threshold:.6f}, "
    f"Pearson>={background_pearson_threshold:.6f}; "
    f"{B_library} measurements -> {B} clusters"
)
for cluster_id, members in enumerate(background_clusters):
    labels = ", ".join(f"Background_{b}" for b in members)
    print(
        f"  background cluster {cluster_id:02d}: representative="
        f"Background_{members[0]}; members={labels}"
    )
print(f"Alignment shifts  : {P} total ({K} isotopes + {B} background)")

# =============================================================================
# LIKELIHOOD UTILITIES
# =============================================================================
def compute_mu(A0, weights, Gamma0, theta, Z, T0, F_iso_energy,
               F_bg_energy, iso_index, energy_template):
    """Expected counts using first-in-cluster representative templates."""
    signal = np.zeros_like(energy_template, dtype=float)
    for j in range(len(A0)):
        if Z[j] == 1 and A0[j] > 0:
            isotope_response = np.zeros_like(energy_template, dtype=float)
            for cluster_id in iso_index[j]:
                isotope_response += weights[cluster_id] * F_iso_energy[cluster_id](
                    energy_template + theta[j]
                )
            signal += A0[j] * isotope_response
    bg = np.zeros_like(energy_template, dtype=float)
    for b in range(len(Gamma0)):
        if Gamma0[b] > 0:
            bg += Gamma0[b] * F_bg_energy[b](
                energy_template + theta[len(A0) + b]
            )
    return np.clip(T0 * (signal + bg), 1e-12, None)


def loglik(Y0, A0, weights, Gamma0, theta, Z, T0, F_iso_energy,
           F_bg_energy, iso_index, energy_template):
    mu = compute_mu(
        A0, weights, Gamma0, theta, Z, T0, F_iso_energy, F_bg_energy,
        iso_index, energy_template
    )
    return np.sum(Y0 * np.log(mu) - mu)

# =============================================================================
# METROPOLIS–HASTINGS UPDATE FUNCTIONS
# (Each returns True on acceptance, False on rejection — for acceptance tracking)
# =============================================================================
def mh_update_A_j(j, Y0, A0, weights, Gamma0, theta, Z, T0,
                  alpha_ig, beta_ig, F_iso_energy, F_bg_energy, step_logA,
                  iso_index, energy_template, rng):
    if Z[j] == 0 or A0[j] <= 0:
        return False
    A_curr  = A0[j]
    ll_curr = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_curr = -(alpha_ig + 1.0) * np.log(A_curr) - beta_ig / A_curr

    A_prop  = A_curr * np.exp(step_logA[j] * rng.normal())
    A0[j]   = A_prop
    ll_prop = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_prop = -(alpha_ig + 1.0) * np.log(A_prop) - beta_ig / A_prop
    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr) + np.log(A_prop / A_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    A0[j] = A_curr
    return False


def mh_update_Gamma_b(b, Y0, A0, weights, Gamma0, theta, Z, T0,
                      s_gamma, F_iso_energy, F_bg_energy, step_gamma,
                      iso_index, energy_template, rng):
    G_curr  = Gamma0[b]
    if G_curr <= 0:
        return False
    ll_curr = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_curr = -0.5 * (G_curr ** 2) / (s_gamma[b] ** 2)

    G_prop  = G_curr + rng.normal(0.0, step_gamma[b])
    if G_prop <= 0:
        return False
    Gamma0[b] = G_prop
    ll_prop = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_prop = -0.5 * (G_prop ** 2) / (s_gamma[b] ** 2)
    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    Gamma0[b] = G_curr
    return False


def mh_update_theta_p(p, Y0, A0, weights, Gamma0, theta, Z, T0,
                      sigma_theta, F_iso_energy, F_bg_energy, step_theta,
                      iso_index, energy_template, rng):
    # When isotope p is absent, theta_p is independent of the likelihood and
    # its full conditional is exactly its Normal prior. Drawing it directly
    # prevents meaningless off-state random-walk modes.
    if p < len(A0) and Z[p] == 0:
        theta[p] = rng.normal(0.0, sigma_theta)
        return True

    theta_curr = theta[p]
    ll_curr = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_curr = -0.5 * (theta_curr / sigma_theta) ** 2

    theta_prop = theta_curr + rng.normal(0.0, step_theta[p])
    theta[p] = theta_prop
    ll_prop = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    lp_prop = -0.5 * (theta_prop / sigma_theta) ** 2
    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    theta[p] = theta_curr
    return False


def mh_flip_Z_j(j, Y0, A0, weights, Gamma0, theta, Z, T0,
                alpha_ig, beta_ig, pi, F_iso_energy, F_bg_energy,
                iso_index, energy_template, rng):
    """Flip Z_j with an activity draw from the IG slab when switching on.

    Because the on-proposal draws A_j from the same IG density used as the
    slab prior, the activity prior/proposal terms cancel in the MH ratio.
    The flip acceptance therefore compares the likelihood and Bernoulli
    inclusion prior only.
    """
    Z_curr  = Z[j]
    A_curr  = A0[j]
    ll_curr = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)

    if Z_curr == 1:
        if A_curr <= 0:
            return False
    lp_Z_curr = Z_curr * np.log(pi[j]) + (1 - Z_curr) * np.log(1 - pi[j])

    if Z_curr == 0:
        Z_prop = 1
        A_prop = 1.0 / rng.gamma(shape=alpha_ig, scale=1.0 / beta_ig)
        A0[j]  = A_prop
    else:
        Z_prop = 0
        A_prop = 0.0
        A0[j]  = 0.0
    Z[j] = Z_prop

    ll_prop = loglik(Y0, A0, weights, Gamma0, theta, Z, T0,
                     F_iso_energy, F_bg_energy, iso_index, energy_template)
    if Z_prop == 1:
        if A_prop <= 0:
            Z[j] = Z_curr; A0[j] = A_curr
            return False
    lp_Z_prop = Z_prop * np.log(pi[j]) + (1 - Z_prop) * np.log(1 - pi[j])
    log_acc   = (ll_prop + lp_Z_prop) - (ll_curr + lp_Z_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    Z[j] = Z_curr; A0[j] = A_curr
    return False


def log_dirichlet_pdf(values, alpha):
    """Log density of a Dirichlet distribution."""
    values = np.asarray(values, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if np.any(values <= 0) or np.any(alpha <= 0):
        return -np.inf
    return (
        gammaln(alpha.sum()) - gammaln(alpha).sum()
        + np.sum((alpha - 1.0) * np.log(values))
    )


def mh_update_weights_j(j, Y0, A0, weights, Gamma0, theta, Z, T0,
                        weight_dirichlet_alpha,
                        weight_proposal_concentration,
                        F_iso_energy, F_bg_energy, iso_index,
                        energy_template, rng):
    """Update simplex weights for the retained templates of isotope j."""
    cols_j = iso_index[j]
    if len(cols_j) <= 1:
        weights[cols_j] = 1.0
        return False

    prior_alpha = weight_dirichlet_alpha * np.ones(len(cols_j))
    if Z[j] == 0:
        # With the isotope absent, weights have their prior full conditional.
        weights[cols_j] = rng.dirichlet(prior_alpha)
        return True

    current = weights[cols_j].copy()
    ll_curr = loglik(
        Y0, A0, weights, Gamma0, theta, Z, T0,
        F_iso_energy, F_bg_energy, iso_index, energy_template
    )
    lp_curr = log_dirichlet_pdf(current, prior_alpha)

    proposal_concentration = weight_proposal_concentration[j]
    forward_alpha = proposal_concentration * current + 1.0
    proposal = rng.dirichlet(forward_alpha)
    weights[cols_j] = proposal
    ll_prop = loglik(
        Y0, A0, weights, Gamma0, theta, Z, T0,
        F_iso_energy, F_bg_energy, iso_index, energy_template
    )
    lp_prop = log_dirichlet_pdf(proposal, prior_alpha)

    reverse_alpha = proposal_concentration * proposal + 1.0
    log_q_forward = log_dirichlet_pdf(proposal, forward_alpha)
    log_q_reverse = log_dirichlet_pdf(current, reverse_alpha)
    log_acc = (
        ll_prop + lp_prop + log_q_reverse
        - ll_curr - lp_curr - log_q_forward
    )

    if np.log(rng.uniform()) < log_acc:
        return True
    weights[cols_j] = current
    return False


# =============================================================================
# INITIALIZATION
# =============================================================================
def init_random(K, B, iso_index, alpha_ig, beta_ig, pi, s_gamma, sigma_theta,
                weight_dirichlet_alpha, rng, min_gamma=1e-6):
    Z0 = rng.binomial(1, pi).astype(int)
    A0 = np.zeros(K)
    for j in range(K):
        if Z0[j] == 1:
            A0[j] = 1.0 / rng.gamma(shape=alpha_ig, scale=1.0 / beta_ig)
    Gamma0 = np.clip(
        np.abs(rng.normal(loc=0.0, scale=s_gamma, size=B)), min_gamma, None
    )
    weights0 = np.zeros(sum(len(cols_j) for cols_j in iso_index), dtype=float)
    for cols_j in iso_index:
        weights0[cols_j] = rng.dirichlet(
            weight_dirichlet_alpha * np.ones(len(cols_j))
        )
    theta0 = rng.normal(loc=0.0, scale=sigma_theta, size=K + B)
    return A0, weights0, Gamma0, theta0, Z0

# =============================================================================
# SINGLE CHAIN RUNNER (with acceptance tracking)
# =============================================================================
def run_single_chain(Y0, T0, F_iso_energy, F_bg_energy, iso_index,
                     energy_template,
                     alpha_ig, beta_ig, pi, s_gamma,
                     sigma_theta, weight_dirichlet_alpha,
                     weight_proposal_concentration,
                     step_logA, step_gamma, step_theta,
                     n_iter, burnin, seed, isotope_names=None,
                     template_names=None,
                     alignment_names=None, verbose=True, progress_every=100):

    rng = np.random.default_rng(seed)
    K_local = len(iso_index)
    M_local = len(F_iso_energy)
    B_local = len(s_gamma)
    P_local = K_local + B_local

    A0, weights, Gamma0, theta0, Z0 = init_random(
        K=K_local, B=B_local, iso_index=iso_index,
        alpha_ig=alpha_ig, beta_ig=beta_ig, pi=pi, s_gamma=s_gamma,
        sigma_theta=sigma_theta,
        weight_dirichlet_alpha=weight_dirichlet_alpha, rng=rng
    )
    theta = theta0.copy()
    Z = Z0.copy()

    if verbose:
        print(f"\n=== Chain seed {seed} initial state ===")
        if isotope_names is not None:
            print(f"{'Isotope':30s}  {'A_init(kBq)':>12}  {'Z':>3}")
            print("-" * 30)
            for j, name in enumerate(isotope_names):
                print(f"{name:30s}  {A0[j]/1e3:>12.4f}  {Z[j]:>3}")
        if template_names is not None:
            print("Weight_init:")
            for m, name in enumerate(template_names):
                print(f"  {name:34s}  {weights[m]:.5f}")
        print(f"Gamma_init: {Gamma0}")
        if alignment_names is not None:
            theta_text = ", ".join(
                f"{name}={theta[p]:.5f} keV"
                for p, name in enumerate(alignment_names)
            )
            print(f"Theta_init: {theta_text}")
        else:
            print(f"Theta_init: {theta}")

    samples_A      = np.zeros((n_iter, K_local))
    samples_weights = np.zeros((n_iter, M_local))
    samples_Gamma  = np.zeros((n_iter, B_local))
    samples_theta  = np.zeros((n_iter, P_local))
    samples_Z      = np.zeros((n_iter, K_local), dtype=int)
    samples_loglik = np.zeros(n_iter)

    # Acceptance counters (post-burn-in only)
    acc_A     = np.zeros(K_local)
    acc_weights = np.zeros(K_local)
    acc_gamma = np.zeros(B_local)
    acc_theta = np.zeros(P_local)
    acc_Z     = np.zeros(K_local)

    for it in range(n_iter):
        for j in range(K_local):
            accepted = int(mh_flip_Z_j(
                j, Y0, A0, weights, Gamma0, theta, Z, T0,
                alpha_ig, beta_ig, pi, F_iso_energy, F_bg_energy,
                iso_index, energy_template, rng
            ))
            if it >= burnin:
                acc_Z[j] += accepted
        for j in range(K_local):
            accepted = int(mh_update_A_j(
                j, Y0, A0, weights, Gamma0, theta, Z, T0,
                alpha_ig, beta_ig, F_iso_energy, F_bg_energy, step_logA,
                iso_index, energy_template, rng
            ))
            if it >= burnin:
                acc_A[j] += accepted
        for j in range(K_local):
            accepted = int(mh_update_weights_j(
                j, Y0, A0, weights, Gamma0, theta, Z, T0,
                weight_dirichlet_alpha, weight_proposal_concentration,
                F_iso_energy, F_bg_energy, iso_index, energy_template, rng
            ))
            if it >= burnin:
                acc_weights[j] += accepted
        for b in range(B_local):
            accepted = int(mh_update_Gamma_b(
                b, Y0, A0, weights, Gamma0, theta, Z, T0, s_gamma,
                F_iso_energy, F_bg_energy, step_gamma, iso_index,
                energy_template, rng
            ))
            if it >= burnin:
                acc_gamma[b] += accepted
        for p in range(P_local):
            accepted = int(mh_update_theta_p(
                p, Y0, A0, weights, Gamma0, theta, Z, T0, sigma_theta,
                F_iso_energy, F_bg_energy, step_theta, iso_index,
                energy_template, rng
            ))
            if it >= burnin:
                acc_theta[p] += accepted

        samples_A[it]      = A0
        samples_weights[it] = weights
        samples_Gamma[it]  = Gamma0
        samples_theta[it]  = theta
        samples_Z[it]      = Z
        samples_loglik[it] = loglik(
            Y0, A0, weights, Gamma0, theta, Z, T0,
            F_iso_energy, F_bg_energy, iso_index, energy_template
        )

        if verbose and ((it + 1) % progress_every == 0 or it == n_iter - 1):
            print(f"  Seed {seed}: iteration {it+1}/{n_iter}")

    n_post = n_iter - burnin
    if verbose:
        print(f"\n  Acceptance rates (post burn-in, {n_post} samples):")
        if isotope_names is not None:
            print(
                f"  {'Isotope':30s}  {'Z flip':>8}  {'A':>8}  "
                f"{'Weights':>8}"
            )
            for j, name in enumerate(isotope_names):
                print(f"  {name:30s}  {acc_Z[j]/n_post:>8.3f}  "
                      f"{acc_A[j]/n_post:>8.3f}  "
                      f"{acc_weights[j]/n_post:>8.3f}")
        print(f"  Gamma: {(acc_gamma / n_post).round(3)}")
        print(f"  Theta: {(acc_theta / n_post).round(3)}")
        print(
            "  Target acceptance: A/weights/θ/γ → 0.20–0.40; "
            "Z flips → 0.10–0.30"
        )

    return {
        "A":         samples_A[burnin:],
        "weights":   samples_weights[burnin:],
        "Gamma":     samples_Gamma[burnin:],
        "theta":     samples_theta[burnin:],
        "Z":         samples_Z[burnin:],
        "loglik":    samples_loglik[burnin:],
        "acc_A":     acc_A / n_post,
        "acc_weights": acc_weights / n_post,
        "acc_gamma": acc_gamma / n_post,
        "acc_theta": acc_theta / n_post,
        "acc_Z":     acc_Z / n_post,
    }

# =============================================================================
# MULTI-CHAIN RUNNER
# =============================================================================
def run_multiple_chains(Y0, T0, F_iso_energy, F_bg_energy, iso_index,
                        energy_template,
                        alpha_ig, beta_ig, pi, s_gamma,
                        sigma_theta, weight_dirichlet_alpha,
                        weight_proposal_concentration,
                        step_logA, step_gamma, step_theta,
                        n_iter=4000, burnin=2000, n_chains=4,
                        base_seed=123, isotope_names=None,
                        template_names=None,
                        alignment_names=None, verbose=True, progress_every=100):

    K_local = len(iso_index)
    M_local = len(F_iso_energy)
    B_local = len(s_gamma)
    n_keep  = n_iter - burnin

    chains_A      = np.zeros((n_chains, n_keep, K_local))
    chains_weights = np.zeros((n_chains, n_keep, M_local))
    chains_Gamma  = np.zeros((n_chains, n_keep, B_local))
    chains_theta  = np.zeros((n_chains, n_keep, K_local + B_local))
    chains_Z      = np.zeros((n_chains, n_keep, K_local), dtype=int)
    chains_loglik = np.zeros((n_chains, n_keep))

    for ch in range(n_chains):
        seed = base_seed + 1000 * ch
        out  = run_single_chain(
            Y0=Y0, T0=T0, F_iso_energy=F_iso_energy,
            F_bg_energy=F_bg_energy, iso_index=iso_index,
            energy_template=energy_template,
            alpha_ig=alpha_ig, beta_ig=beta_ig, pi=pi, s_gamma=s_gamma,
            sigma_theta=sigma_theta,
            weight_dirichlet_alpha=weight_dirichlet_alpha,
            weight_proposal_concentration=weight_proposal_concentration,
            step_logA=step_logA,
            step_gamma=step_gamma, step_theta=step_theta,
            n_iter=n_iter, burnin=burnin, seed=seed,
            isotope_names=isotope_names, template_names=template_names,
            alignment_names=alignment_names,
            verbose=verbose,
            progress_every=progress_every
        )
        chains_A[ch]      = out["A"]
        chains_weights[ch] = out["weights"]
        chains_Gamma[ch]  = out["Gamma"]
        chains_theta[ch]  = out["theta"]
        chains_Z[ch]      = out["Z"]
        chains_loglik[ch] = out["loglik"]

    return {
        "A": chains_A, "weights": chains_weights, "Gamma": chains_Gamma,
        "theta": chains_theta,
        "Z": chains_Z, "loglik": chains_loglik,
    }

# =============================================================================
# RUN MCMC   timer wraps the full sampling block
# =============================================================================
n_chains = args.n_chains
n_iter   = args.n_iter
burnin   = args.burnin
if n_chains < 1 or n_iter < 1 or not (0 <= burnin < n_iter):
    raise ValueError(
        "Require n_chains >= 1, n_iter >= 1, and 0 <= burnin < n_iter."
    )

print(f"\n{'='*60}")
print(f"Starting MCMC: {n_chains} chains × {n_iter} iterations (burn-in {burnin})")
print(f"{'='*60}")

t_wall_start = time.time()

multi_out = run_multiple_chains(
    Y0=Y0, T0=T0, F_iso_energy=F_template_energy, F_bg_energy=F_bg_energy,
    iso_index=model_iso_index, energy_template=energy_template,
    alpha_ig=alpha_ig, beta_ig=beta_ig, pi=pi, s_gamma=s_gamma,
    sigma_theta=sigma_theta,
    weight_dirichlet_alpha=weight_dirichlet_alpha,
    weight_proposal_concentration=weight_proposal_concentration,
    step_logA=step_logA,
    step_gamma=step_gamma, step_theta=step_theta,
    n_iter=n_iter, burnin=burnin, n_chains=n_chains,
    base_seed=123, isotope_names=model_isotope_names,
    template_names=template_labels,
    alignment_names=alignment_labels,
    verbose=True,
    progress_every=100
)

elapsed = time.time() - t_wall_start
h, rem  = divmod(int(elapsed), 3600)
m, s    = divmod(rem, 60)
print(f"\nTotal wall time: {h:02d}:{m:02d}:{s:02d}")

# =============================================================================
# SAVE CHAINS TO DISK
# =============================================================================
chains_A      = multi_out["A"]
chains_weights = multi_out["weights"]
chains_Gamma  = multi_out["Gamma"]
chains_theta  = multi_out["theta"]
chains_Z      = multi_out["Z"]
chains_loglik = multi_out["loglik"]

np.save(save_dir / "chains_A.npy",      chains_A)
np.save(save_dir / "chains_weights.npy", chains_weights)
np.save(save_dir / "chains_Gamma.npy",  chains_Gamma)
np.save(save_dir / "chains_theta.npy",  chains_theta)
np.save(save_dir / "chains_Z.npy",      chains_Z)
np.save(save_dir / "chains_loglik.npy", chains_loglik)
np.save(save_dir / "library_template_labels.npy", library_template_labels)
np.save(save_dir / "template_labels.npy", template_labels)
np.save(save_dir / "cluster_representative_templates.npy", X_template)
np.save(
    save_dir / "cluster_representative_candidate_indices.npy",
    representative_candidate_idx,
)
np.save(save_dir / "background_representative_templates.npy", X_background)
np.save(
    save_dir / "background_representative_indices.npy",
    background_representative_idx,
)
np.save(save_dir / "isotope_labels.npy", model_isotope_names)
np.save(save_dir / "alignment_labels.npy", alignment_labels)
print(f"Chains saved to: {save_dir}")

# =============================================================================
# POOL CHAINS
# =============================================================================
chains_A_iso = chains_A
chains_Z_iso = chains_Z
np.save(save_dir / "chains_A_iso.npy", chains_A_iso)
np.save(save_dir / "chains_Z_iso.npy", chains_Z_iso)

A_post          = chains_A.reshape(-1, K)
weights_post    = chains_weights.reshape(-1, M)
Gamma_post      = chains_Gamma.reshape(-1, B)
theta_post      = chains_theta.reshape(-1, P)
Z_post          = chains_Z.reshape(-1, K)

A_post_kBq = A_post / 1e3
incl_prob  = Z_post.mean(axis=0)

print("\nPosterior inclusion probabilities — isotopes:")
for j, lab in enumerate(model_isotope_names):
    n_on  = int((Z_post[:, j] == 1).sum())
    n_off = int((Z_post[:, j] == 0).sum())
    print(
        f"  {lab:6s}: P(Z=1|Y) = {incl_prob[j]:.4f}  "
        f"(n_on={n_on}, n_off={n_off})"
    )

weight_summary_rows = []
print("\nPosterior summaries — template-cluster weights:")
for m, label in enumerate(template_labels):
    mean_w = float(weights_post[:, m].mean())
    lo_w, hi_w = np.percentile(weights_post[:, m], [2.5, 97.5])
    isotope = model_isotope_names[iso_of_template_model[m]]
    print(
        f"  w[{isotope}, m={m:02d}] {label:34s}: "
        f"mean={mean_w:.4f}, 95% CI=[{lo_w:.4f}, {hi_w:.4f}]"
    )
    weight_summary_rows.append({
        "Cluster_index": m,
        "Isotope": isotope,
        "Template": label,
        "Cluster_members": "; ".join(
            candidate_template_labels[candidate_id]
            for candidate_id in cluster_members[m]
        ),
        "Posterior_mean_weight": mean_w,
        "Posterior_95CI_lo": lo_w,
        "Posterior_95CI_hi": hi_w,
    })
weight_summary = pd.DataFrame(weight_summary_rows)
weight_summary.to_csv(
    save_dir / "posterior_template_weights.csv", index=False
)
print(
    f"Template-weight table saved: "
    f"{save_dir / 'posterior_template_weights.csv'}"
)

# =============================================================================
# [2] CONVERGENCE DIAGNOSTICS: R-HAT AND ESS VIA ARVIZ
# =============================================================================
pip_threshold  = 0.5
selected_idx   = np.where(incl_prob >= pip_threshold)[0]
active_labels  = model_isotope_names[selected_idx]
background_shift_idx = K + np.arange(B, dtype=int)
diagnostic_weight_idx = np.array([
    m for m in range(M)
    if len(model_iso_index[iso_of_template_model[m]]) > 1
], dtype=int)

# Build ArviZ InferenceData — shape required: (n_chains, n_draws, ...)
idata_dict = {
    **{f"A_{model_isotope_names[j]}": chains_A_iso[:, :, j]
       for j in selected_idx},
    **{f"theta_{model_isotope_names[j]}": chains_theta[:, :, j]
       for j in selected_idx},
    **{f"w_m{m}": chains_weights[:, :, m]
       for m in diagnostic_weight_idx},
    **{f"theta_bg{b}": chains_theta[:, :, K + b] for b in range(B)},
    **{f"gamma_{b}": chains_Gamma[:, :, b] for b in range(B)},
}
try:
    idata = az.from_dict(posterior=idata_dict)
except TypeError:
    idata = az.from_dict({"posterior": idata_dict})

print(f"\n{'='*60}")
print("Convergence diagnostics (active isotopes, R-hat and ESS)")
print(f"{'='*60}")
print(f"{'Parameter':<18}  {'R-hat':>8}  {'ESS bulk':>10}  {'ESS tail':>10}  {'Status'}")
print("-" * 60)
for iso in active_labels:
    var    = f"A_{iso}"
    rhat   = float(az.rhat(idata)[var].values)
    ess_b  = float(az.ess(idata, method="bulk")[var].values)
    ess_t  = float(az.ess(idata, method="tail")[var].values)
    status = "✓ OK" if rhat < 1.01 and ess_b > 400 else "⚠ CHECK"
    print(f"  {var:<16}  {rhat:>8.4f}  {ess_b:>10.1f}  {ess_t:>10.1f}  {status}")

# Isotope-specific energy shifts
for j in selected_idx:
    var    = f"theta_{model_isotope_names[j]}"
    rhat   = float(az.rhat(idata)[var].values)
    ess_b  = float(az.ess(idata, method="bulk")[var].values)
    ess_t  = float(az.ess(idata, method="tail")[var].values)
    status = "✓ OK" if rhat < 1.01 and ess_b > 400 else "⚠ CHECK"
    print(f"  {var:<16}  {rhat:>8.4f}  {ess_b:>10.1f}  {ess_t:>10.1f}  {status}")

# Retained-template nuisance weights
for m in diagnostic_weight_idx:
    var    = f"w_m{m}"
    rhat   = float(az.rhat(idata)[var].values)
    ess_b  = float(az.ess(idata, method="bulk")[var].values)
    ess_t  = float(az.ess(idata, method="tail")[var].values)
    status = "✓ OK" if rhat < 1.01 and ess_b > 400 else "⚠ CHECK"
    print(f"  {var:<16}  {rhat:>8.4f}  {ess_b:>10.1f}  {ess_t:>10.1f}  {status}")

# Background energy shifts
for b in range(B):
    var    = f"theta_bg{b}"
    rhat   = float(az.rhat(idata)[var].values)
    ess_b  = float(az.ess(idata, method="bulk")[var].values)
    ess_t  = float(az.ess(idata, method="tail")[var].values)
    status = "✓ OK" if rhat < 1.01 and ess_b > 400 else "⚠ CHECK"
    print(f"  {var:<16}  {rhat:>8.4f}  {ess_b:>10.1f}  {ess_t:>10.1f}  {status}")

# Background coefficients
for b in range(B):
    var    = f"gamma_{b}"
    rhat   = float(az.rhat(idata)[var].values)
    ess_b  = float(az.ess(idata, method="bulk")[var].values)
    ess_t  = float(az.ess(idata, method="tail")[var].values)
    status = "✓ OK" if rhat < 1.01 and ess_b > 400 else "⚠ CHECK"
    print(f"  {var:<16}  {rhat:>8.4f}  {ess_b:>10.1f}  {ess_t:>10.1f}  {status}")

print(f"\nRule: R-hat < 1.01 and ESS > 400 indicate convergence.")

# =============================================================================
# POSTERIOR SUMMARIES
# =============================================================================
def summarize(name, samples, labels, scale=1.0, unit=""):
    means  = samples.mean(axis=0) * scale
    lower  = np.percentile(samples, 2.5,  axis=0) * scale
    upper  = np.percentile(samples, 97.5, axis=0) * scale
    for k, lab in enumerate(labels):
        print(f"  {name} {lab:8s}: mean = {means[k]:.4g}{unit}, "
              f"95% CI = [{lower[k]:.4g}, {upper[k]:.4g}]{unit}")

print("\nPosterior summaries — activities (kBq):")
summarize("A", A_post, model_isotope_names, scale=1e-3, unit=" kBq")

print("\nPosterior summaries — background coefficients:")
summarize("γ", Gamma_post, ["total"])

print("\nPosterior summaries — isotope-specific energy shifts:")
summarize(
    "θ", theta_post[:, selected_idx],
    model_isotope_names[selected_idx], unit=" keV"
)

print("\nPosterior summaries — background energy shifts:")
summarize(
    "θ", theta_post[:, background_shift_idx], background_shift_labels,
    unit=" keV"
)

# =============================================================================
# REFERENCE ACTIVITY COMPARISON — PEAKANALYSIS_G8 FROM FILE 021
# =============================================================================
reference_table = pd.read_csv(base / "021_A_REF_main.csv")
g8_rows = reference_table.loc[
    reference_table["MeasurementSeriesID_613"].astype(str).eq(str(mix0["MSID"]))
    & reference_table["REF_type"].astype(str).eq("PeakAnalysis_G8")
].copy()

if g8_rows.empty:
    raise ValueError(
        f"No PeakAnalysis_G8 references found in 021_A_REF_main.csv "
        f"for {mix0['MSID']}."
    )

required_g8_columns = ["Nuclide", "t_ref", "A_Bq", "uA_Bq", "REF_file"]
if g8_rows[required_g8_columns].isna().any().any():
    raise ValueError(
        f"Incomplete PeakAnalysis_G8 reference data for {mix0['MSID']}."
    )

g8_rows["A_Bq"] = pd.to_numeric(g8_rows["A_Bq"], errors="raise")
g8_rows["uA_Bq"] = pd.to_numeric(g8_rows["uA_Bq"], errors="raise")
g8_rows["t_ref"] = pd.to_datetime(g8_rows["t_ref"], errors="raise")
if g8_rows.duplicated(["Nuclide", "Subset", "REF_file"]).any():
    raise ValueError(
        f"Duplicate PeakAnalysis_G8 component rows found for {mix0['MSID']}."
    )

reference_dates = g8_rows["t_ref"].drop_duplicates()
if len(reference_dates) != 1:
    raise ValueError(
        f"PeakAnalysis_G8 rows for {mix0['MSID']} do not share one "
        f"reference date: {reference_dates.astype(str).tolist()}"
    )
reference_date = pd.Timestamp(reference_dates.iloc[0])

# Some mixtures contain multiple physical subsets. Their activities add, and
# the corresponding standard uncertainties are combined in quadrature.
g8_reference = {}
for iso, iso_rows in g8_rows.groupby("Nuclide", sort=False):
    g8_reference[str(iso)] = {
        "A_Bq": float(iso_rows["A_Bq"].sum()),
        "uA_Bq": float(np.sqrt(np.sum(iso_rows["uA_Bq"].to_numpy() ** 2))),
        "n_components": int(len(iso_rows)),
        "reference_files": "; ".join(
            sorted(iso_rows["REF_file"].astype(str).unique())
        ),
    }

expected_g8_isotopes = set(
    tmls.loc[
        tmls["MeasurementSeriesID_613"].astype(str).eq(str(mix0["MSID"])),
        "Nuclide",
    ].dropna().astype(str)
)
if set(g8_reference) != expected_g8_isotopes:
    raise ValueError(
        f"PeakAnalysis_G8 isotope mismatch for {mix0['MSID']}: expected "
        f"{sorted(expected_g8_isotopes)}, found {sorted(g8_reference)}."
    )

run_filename = str(mix0["Filename"][r_run])
run_start = pd.Timestamp(
    spectra.loc[spectra["Filename"].astype(str) == run_filename, "t_start"].iloc[0]
)
elapsed_seconds_to_reference = (reference_date - run_start).total_seconds()

half_lives = pd.read_csv(base / "05_Half_lives.csv")
half_life_seconds = dict(zip(
    half_lives["Nuclide"].astype(str), half_lives["HalfLife_s"].astype(float)
))
iso_idx = {name: i for i, name in enumerate(model_isotope_names)}
unknown_g8_isotopes = sorted(set(g8_reference) - set(iso_idx))
if unknown_g8_isotopes:
    raise ValueError(
        f"PeakAnalysis_G8 contains isotopes outside the candidate model: "
        f"{unknown_g8_isotopes}"
    )

comparison_width = 93
print(f"\n{'='*comparison_width}")
print(f"PeakAnalysis_G8 comparison — mixture {i_mix}, run {r_run}: {run_filename}")
print(f"G8 references loaded from 021_A_REF_main.csv for {mix0['MSID']}")
if any(ref["n_components"] > 1 for ref in g8_reference.values()):
    print("Multiple G8 subset activities summed; uncertainties combined in quadrature")
print(f"Posterior activities decay-corrected from {run_start} to {reference_date}")
print(f"{'='*comparison_width}")
print(f"{'Isotope':<8} {'Post.Mean':>10} {'99% CI (kBq)':>22} "
      f"{'G8 (kBq)':>10} {'uG8':>8} "
      f"{'Bias%':>8} {'En':>8} {'Cover99':>7}")
print("-" * comparison_width)

comparison_rows = []
for iso, ref in g8_reference.items():
    j = iso_idx[iso]
    decay_factor = np.exp(
        -np.log(2.0) * elapsed_seconds_to_reference / half_life_seconds[iso]
    )
    posterior_at_reference = A_post[:, j] * decay_factor
    pm_Bq = posterior_at_reference.mean()
    lo_Bq, hi_Bq = np.percentile(posterior_at_reference, [2.5, 97.5])
    lo99_Bq, hi99_Bq = np.percentile(posterior_at_reference, [0.5, 99.5])
    U_post_Bq = (hi_Bq - lo_Bq) / 2.0

    A_ref = ref["A_Bq"]
    uA_ref = ref["uA_Bq"]
    U_ref_Bq = 2.0 * uA_ref
    rel_bias = (pm_Bq - A_ref) / A_ref * 100.0
    En = (pm_Bq - A_ref) / np.sqrt(U_post_Bq**2 + U_ref_Bq**2)
    covered = lo99_Bq <= A_ref <= hi99_Bq

    print(f"{iso:<8} {pm_Bq/1e3:>10.4f} "
          f"[{lo99_Bq/1e3:.4f}, {hi99_Bq/1e3:.4f}] "
          f"{A_ref/1e3:>10.4f} {uA_ref/1e3:>8.4f} "
          f"{rel_bias:>8.2f} {En:>8.3f} {str(covered):>7}")

    comparison_rows.append({
        "Filename": run_filename,
        "Reference_date": reference_date,
        "Isotope": iso,
        "Decay_factor_to_reference": decay_factor,
        "Post_mean_kBq": pm_Bq / 1e3,
        "CI99_lo_kBq": lo99_Bq / 1e3,
        "CI99_hi_kBq": hi99_Bq / 1e3,
        "PIP": incl_prob[j],
        "Reference_type": "PeakAnalysis_G8",
        "Ref_G8_kBq": A_ref / 1e3,
        "uRef_G8_k1_kBq": uA_ref / 1e3,
        "G8_component_count": ref["n_components"],
        "G8_reference_files": ref["reference_files"],
        "Rel_bias_pct": rel_bias,
        "En": En,
        "Covered_99pct": covered,
    })

df_comp = pd.DataFrame(comparison_rows)
df_comp.to_csv(save_dir / "reference_comparison_G8.csv", index=False)
print(f"\nComparison table saved: {save_dir / 'reference_comparison_G8.csv'}")

# =============================================================================
# PLOTTING
# =============================================================================
# Figures are generated from the saved posterior draws by plotting.py. This
# keeps plotting changes independent of the MCMC run.
print(f"\n{'='*60}")
print(f"All numerical outputs saved to: {save_dir}")
print("Run plotting.py to generate or update figures from these chains.")
print(f"{'='*60}")
