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

# select mixture and run
i_mix = 1
r_run = 0

mix0 = mls_spectra[i_mix]
Y0 = mix0["Counts"][r_run, :]
T0 = mix0["Livetime"][r_run]

# setting hyper-parameters
alpha_ig = 4
beta_ig = 3394

pi = 0.5 * np.ones(K)
sigma_theta = 0.05
s_gamma = 0.5 * np.ones(B)

# stepsize for random walk
step_logA  = 0.1
step_gamma = 0.1
step_theta = 0.1

rng = np.random.default_rng()

# define the poisson mean term
def compute_mu_single(A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg):
    signal = np.zeros_like(channels, dtype=float)

    for j in range(K):
        if Z[j] == 1 and A0[j] > 0:
            signal += A0[j] * F_iso_avg[j](channels + theta[j])

    bg = np.zeros_like(channels, dtype=float)
    for b in range(B):
        if Gamma0[b] > 0:
            bg += Gamma0[b] * F_bg[b](channels)

    mu = T0 * (signal + bg)
    return np.clip(mu, 1e-12, None)

# log-likelihood
def loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg):
    mu = compute_mu_single(A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    return (Y0 * np.log(mu) - mu).sum()

# update A_j
def mh_update_A_j(j, Y0, A0, Gamma0, theta, Z, T0,
                  alpha_ig, beta_ig,
                  F_iso_avg, F_bg,
                  step_logA):

    if Z[j] == 0:
        return False

    A_curr = A0[j]
    if A_curr <= 0:
        return False

    ll_curr = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_curr = -(alpha_ig + 1.0) * np.log(A_curr) - (beta_ig / A_curr)
    z = rng.normal(0.0, 1.0)
    A_prop = A_curr * np.exp(step_logA * z)

    A0[j] = A_prop
    ll_prop = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_prop = -(alpha_ig + 1.0) * np.log(A_prop) - (beta_ig / A_prop)
    log_qcorr = np.log(A_prop / A_curr)

    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr) + log_qcorr

    if np.log(rng.uniform()) < log_acc:
        return True
    else:
        A0[j] = A_curr
        return False


# update Gamma_b
def mh_update_Gamma_b(b, Y0, A0, Gamma0, theta, Z, T0,
                      s_gamma,
                      F_iso_avg, F_bg,
                      step_gamma):
    G_curr = Gamma0[b]
    if G_curr <= 0:
        return False

    ll_curr = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_curr = -0.5 * (G_curr**2) / (s_gamma[b]**2)

    G_prop = G_curr + rng.normal(0.0, step_gamma)
    if G_prop <= 0:
        return False

    Gamma0[b] = G_prop
    ll_prop = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_prop = -0.5 * (G_prop**2) / (s_gamma[b]**2)

    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    else:
        Gamma0[b] = G_curr
        return False


# update theta_j
def mh_update_theta_j(j, Y0, A0, Gamma0, theta, Z, T0,
                      sigma_theta,
                      F_iso_avg, F_bg,
                      step_theta):
    th_curr = theta[j]

    ll_curr = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_curr = - (th_curr**2) / (2 * sigma_theta**2)

    th_prop = th_curr + rng.normal(0.0, step_theta)
    if not (-2.0 <= th_prop <= 2.0):
        return False

    theta[j] = th_prop
    ll_prop = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_prop = - (th_prop**2) / (2 * sigma_theta**2)

    log_acc = (ll_prop + lp_prop) - (ll_curr + lp_curr)

    if np.log(rng.uniform()) < log_acc:
        return True
    else:
        theta[j] = th_curr
        return False

# flip Z_j
def mh_flip_Z_j(j, Y0, A0, Gamma0, theta, Z, T0,
                alpha_ig, beta_ig, pi,
                F_iso_avg, F_bg,
                step_logA):
    Z_curr = Z[j]
    A_curr = A0[j]

    ll_curr = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_A_curr = 0.0
    if Z_curr == 1:
        if A_curr <= 0:
            return False
        lp_A_curr = -(alpha_ig + 1.0) * np.log(A_curr) - (beta_ig / A_curr)

    lp_Z_curr = Z_curr * np.log(pi[j]) + (1 - Z_curr) * np.log(1 - pi[j])

    log_qcorr = 0.0

    if Z_curr == 0:
        Z_prop = 1

        A_center = max(A_curr, 1e-8)
        z = rng.normal(0.0, 1.0)
        A_prop = A_center * np.exp(step_logA * z)

        A0[j] = A_prop
        log_qcorr = np.log(A_prop / A_center)
    else:
        Z_prop = 0
        A_prop = 0.0
        A0[j] = 0.0
        log_qcorr = 0.0

    Z[j] = Z_prop

    ll_prop = loglik_single(Y0, A0, Gamma0, theta, Z, T0, F_iso_avg, F_bg)
    lp_A_prop = 0.0
    if Z_prop == 1:
        if A_prop <= 0:
            Z[j] = Z_curr
            A0[j] = A_curr
            return False
        lp_A_prop = -(alpha_ig + 1.0) * np.log(A_prop) - (beta_ig / A_prop)

    lp_Z_prop = Z_prop * np.log(pi[j]) + (1 - Z_prop) * np.log(1 - pi[j])

    log_acc = (ll_prop + lp_A_prop + lp_Z_prop) - (ll_curr + lp_A_curr + lp_Z_curr) + log_qcorr

    if np.log(rng.uniform()) < log_acc:
        return True
    else:
        Z[j] = Z_curr
        A0[j] = A_curr
        return False

# Metropolis-Hastings sweep for single spike-and-slab
def mh_sweep_single_spike_slab(Y0, A0, Gamma0, theta, Z, T0,
                               alpha_ig, beta_ig, pi, s_gamma, sigma_theta,
                               F_iso_avg, F_bg,
                               step_logA, step_gamma, step_theta):

    K_local = A0.size
    B_local = Gamma0.size

    # Flip Z
    for j in range(K_local):
        mh_flip_Z_j(j, Y0, A0, Gamma0, theta, Z, T0,
                    alpha_ig, beta_ig, pi,
                    F_iso_avg, F_bg,
                    step_logA=step_logA)

    # Update A for active isotopes
    for j in range(K_local):
        mh_update_A_j(j, Y0, A0, Gamma0, theta, Z, T0,
                      alpha_ig, beta_ig,
                      F_iso_avg, F_bg,
                      step_logA)

    # Update background coefficients
    for b in range(B_local):
        mh_update_Gamma_b(b, Y0, A0, Gamma0, theta, Z, T0,
                          s_gamma,
                          F_iso_avg, F_bg,
                          step_gamma)

    # Update shifts
    for j in range(K_local):
        mh_update_theta_j(j, Y0, A0, Gamma0, theta, Z, T0,
                          sigma_theta,
                          F_iso_avg, F_bg,
                          step_theta)

        
# EB for initial points
def init_from_nnls(Y0, T0, X_iso_avg, X_bg, theta0=None, F_iso_avg=None, F_bg=None, channels=None):
    C, K = X_iso_avg.shape
    B = X_bg.shape[1]

    if theta0 is None:
        theta0 = np.zeros(K)

    X_design = np.zeros((C, K + B))
    X_design[:, :K] = T0 * X_iso_avg
    X_design[:, K:] = T0 * X_bg

    coeffs, _ = nnls(X_design, Y0)
    A0 = coeffs[:K]
    Gamma0 = coeffs[K:]

    return A0, Gamma0, theta0

A0, Gamma0, theta0 = init_from_nnls(Y0, T0, X_iso_avg, X_bg)

# initialize Z based on threshold
tau_A_init = max(1e-2, 1e-3 * A0.max())
Z0 = (A0 > tau_A_init).astype(int)

A0 = np.clip(A0, 1e-8, None)
Gamma0 = np.clip(Gamma0, 1e-8, None)
theta  = theta0.copy()
Z      = Z0.copy()

print("\n=== Empirical Bayes Initial Values ===")
print(f"Mixture MSID: {mix0['MSID']}")
print(f"Run index: {r_run}\n")

print("Isotope    A_init (kBq)   Z_init")
print("--------------------------------")
for j, name in enumerate(iso_names):
    print(f"{name:8s}  {A0[j]/1e3:10.4f}      {Z[j]}")

print("\nBackground initial values:")
for b in range(len(Gamma0)):
    print(f"Background {b}: Gamma_init = {Gamma0[b]:.4g}")

print("A0 (Bq) min/max:", A0.min(), A0.max())
print("Gamma0 min/max:", Gamma0.min(), Gamma0.max())
print("tau_A_init:", tau_A_init)
print("Z0 counts on:", Z0.sum(), "off:", (1-Z0).sum())

# Metropolis-Hastings sampling
n_iter = 5000
burnin = 1000

samples_A     = np.zeros((n_iter, K))
samples_Gamma = np.zeros((n_iter, B))
samples_theta = np.zeros((n_iter, K))
samples_Z     = np.zeros((n_iter, K), dtype=int)

for it in range(n_iter):
    mh_sweep_single_spike_slab(
        Y0, A0, Gamma0, theta, Z, T0,
        alpha_ig, beta_ig, pi, s_gamma, sigma_theta,
        F_iso_avg, F_bg,
        step_logA, step_gamma, step_theta
    )

    samples_A[it]     = A0
    samples_Gamma[it] = Gamma0
    samples_theta[it] = theta
    samples_Z[it]     = Z

    if (it + 1) % 500 == 0:
        print(f"Iteration {it + 1}")


# posterior draws
A_post      = samples_A[burnin:, :]
A_post_kBq  = A_post / 1e3 
Gamma_post  = samples_Gamma[burnin:, :]
theta_post  = samples_theta[burnin:, :]
Z_post      = samples_Z[burnin:, :]
n_post      = A_post.shape[0]
iso_labels  = iso_names

# --- A_j trace plot (kBq) ---
plt.figure(figsize=(12, 6))
for j in range(K):
    plt.plot(A_post_kBq[:, j], alpha=0.5, label=iso_labels[j])
plt.xlabel("Iteration")
plt.ylabel("Activity $A_j$ (kBq)")
plt.title("Trace of activities")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# --- Gamma_b trace plot ---
plt.figure(figsize=(8, 4))
for b in range(B):
    plt.plot(Gamma_post[:, b], alpha=0.7, label=f"Gamma_{b}")
plt.xlabel("Iteration")
plt.ylabel("Background scale")
plt.title("Trace of background coefficients")
plt.legend()
plt.tight_layout()
plt.show()

# --- theta_j trace plot ---
plt.figure(figsize=(12, 6))
for j in range(K):
    plt.plot(theta_post[:, j], alpha=0.5, label=iso_labels[j])
plt.xlabel("Iteration")
plt.ylabel("Theta_j (channels)")
plt.title("Trace of energy shifts")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# --- posterior summaries ---
def summarize(name, samples, labels):
    means = samples.mean(axis=0)
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    for k, lab in enumerate(labels):
        print(f"{name} {lab:8s}: mean = {means[k]:.3g}, 95% CI = [{lower[k]:.3g}, {upper[k]:.3g}]")

print("Posterior summaries (activities):")
summarize("A (kBq)", A_post_kBq, iso_labels)

print("\nPosterior summaries (backgrounds):")
summarize("Gamma", Gamma_post, [f"b{b}" for b in range(B)])

print("\nPosterior summaries (shifts):")
summarize("theta", theta_post, iso_labels)

print("\nPosterior inclusion probabilities:")
incl_prob = Z_post.mean(axis=0)
for j, lab in enumerate(iso_labels):
    print(f"{lab:6s}: P(Z=1 | Y) ≈ {incl_prob[j]:.3f}")

# --- posterior mean/median fit ---
def compute_mu_from_params(A_j, Gamma_b, theta_j, T0, F_iso_avg, F_bg):
    signal = np.zeros(channels.size, dtype=float)
    for j in range(K):
        if A_j[j] > 0:
            signal += A_j[j] * F_iso_avg[j](channels + theta_j[j])

    bg = np.zeros(channels.size, dtype=float)
    for b in range(B):
        if Gamma_b[b] > 0:
            bg += Gamma_b[b] * F_bg[b](channels)

    mu = T0 * (signal + bg)
    return np.clip(mu, 1e-12, None)

# posterior mean fit overlayed on spectra (log scale)
A_mean       = A_post.mean(axis=0)
Gamma_mean   = Gamma_post.mean(axis=0)
theta_mean   = theta_post.mean(axis=0)
mu_mean   = compute_mu_from_params(A_mean,   Gamma_mean,   theta_mean,   T0, F_iso_avg, F_bg)
ecal_run0 = mix0["Ecal"][r_run, :]
a_cal, b_cal = float(ecal_run0[0]), float(ecal_run0[1])
energy = a_cal + b_cal * channels

rate_obs        = Y0 / T0
rate_fit_mean   = mu_mean / T0

plt.figure(figsize=(10, 4))
plt.step(energy, rate_obs, where="mid", label="Data", alpha=0.7)
plt.step(energy, rate_fit_mean, where="mid", label="Posterior mean", alpha=0.7)
plt.yscale("log")
plt.xlabel("Energy (keV)")
plt.ylabel("Count rate (s$^{-1}$)")
plt.title("Count rate vs energy (log scale)")
plt.legend()
plt.tight_layout()
plt.show()
