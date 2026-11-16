"""Bayesian gamma-spectrum analysis with background-corrected templates.

Place this file beside the input tables. Run numbers are ONE-based.

    python gamma_spec_bayes.py --mixture 849-23-P058-111 --run 1
    python gamma_spec_bayes.py --mixture all --run 1
    python gamma_spec_bayes.py --mixture all --run all

Requires numpy, scipy, pandas, openpyxl, arviz, and matplotlib.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls
from scipy.special import gammaln
from scipy.stats import gamma, norm


# Settings define one model and sampling procedure for every mixture/run.
# Proposal widths are numerical tuning parameters, not prior widths.
ISOTOPES = ("Na22", "Mn54", "Co57", "Co60", "Zn65", "Y88", "Cd109",
            "Cs134", "Cs137", "Pb210", "Am241")


@dataclass
class Settings:
    energy_min: float = 20.0
    energy_max: float = 2800.0
    correction_background: str = "849-NE-23-1418"
    isotope_similarity: float = 0.999
    background_similarity: float = 0.995
    slab_shape: float = 2.5
    slab_quantile: float = 0.005
    detection_alpha: float = 0.05
    detection_beta: float = 0.05
    inclusion_probability: float = 0.5
    shift_prior_sd: float = 0.5
    background_prior_sd: float = 0.5
    broad_shift_probability: float = 0.20
    broad_shift_sd: float = 0.35
    log_activity_step: float = 0.01
    log_background_step: float = 0.004
    isotope_shift_step: float = 0.006
    background_shift_step: float = 0.001
    weight_concentration: float = 2000.0
    adapt_interval: int = 200
    target_acceptance: float = 0.30
    chains: int = 4
    iterations: int = 20000
    burnin: int = 10000
    seed: int = 123


# Load spectra by filename so counts, livetimes, and energy calibrations match.
# Mixture reference activities are used only in the final comparison.
@dataclass
class Spectrum:
    filename: str
    series: str
    counts: np.ndarray
    livetime: float
    calibration: np.ndarray
    start: pd.Timestamp

    @property
    def energy(self):
        b0, b1 = self.calibration
        return b0 + b1 * np.arange(1, len(self.counts) + 1)


class Dataset:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        spectra = pd.read_excel(self.directory / "04_DATA_spectra.xlsx")
        calibrations = pd.read_excel(self.directory / "042_DATA_ECal.xlsx")
        templates = pd.read_excel(self.directory / "03_Templates_A_REF.xlsx")
        mixtures = pd.read_excel(self.directory / "02_A_REF.xlsx")
        if spectra.Filename.duplicated().any() or calibrations.Filename.duplicated().any():
            raise ValueError("Spectrum and calibration filenames must be unique.")
        calibrations = calibrations.set_index("Filename")
        self.spectra = {}
        for i, row in spectra.iloc[:, :4].iterrows():
            calibration = calibrations.loc[row.Filename, ["Ecal_b0_GX", "Ecal_b1_GX"]].to_numpy(float)
            counts = spectra.iloc[i, 4:].to_numpy(float)
            if (not np.isfinite(counts).all() or np.any(counts < 0)
                    or not np.isfinite(calibration).all() or calibration[1] <= 0
                    or not np.isfinite(row.t_live_s) or row.t_live_s <= 0):
                raise ValueError(f"Invalid counts, livetime, or calibration: {row.Filename}")
            self.spectra[row.Filename] = Spectrum(
                row.Filename, row.MeasurementSeriesID_613, counts,
                float(row.t_live_s), calibration, pd.Timestamp(row.t_start))
        self.mixtures = {
            str(series): sorted(group.Filename.unique())
            for series, group in mixtures.groupby("MeasurementSeriesID_613", sort=False)
        }
        self.backgrounds = {
            series: [s for s in self.spectra.values() if s.series == series]
            for series in sorted({s.series for s in self.spectra.values() if "-NE-" in s.series})
        }
        reference_background = self.backgrounds[settings.correction_background]
        self.templates = []
        for isotope in ISOTOPES:
            rows = templates.loc[templates.Nuclide == isotope]
            if rows.empty:
                raise ValueError(f"No calibration templates for {isotope}.")
            for series, group in rows.groupby("MeasurementSeriesID_613", sort=True):
                runs = []
                for _, row in group.iterrows():
                    spectrum = self.spectra[row.Filename]
                    if not np.isfinite(row.A_Bq) or row.A_Bq <= 0:
                        raise ValueError(f"Invalid calibration activity: {row.Filename}")
                    bg_rate = pooled_background(reference_background, spectrum.energy)
                    # Conditional Poisson MLE for a nonnegative native-channel source rate.
                    response = np.maximum(spectrum.counts / spectrum.livetime - bg_rate, 0) / row.A_Bq
                    runs.append((spectrum, response))
                self.templates.append((isotope, f"{isotope}_{series}", runs))


# Subtract background on each calibration run's native grid, then register
# its cps/Bq response and average runs by livetime. Backgrounds retain cps units.
def register(values, calibration, target_energy):
    b0, b1 = calibration
    native_energy = b0 + b1 * np.arange(1, len(values) + 1)
    registered = PchipInterpolator(native_energy, values, extrapolate=True)(target_energy)
    registered *= np.median(np.diff(target_energy)) / b1
    return np.maximum(registered, 0)


def pooled_background(spectra, energy):
    times = np.array([s.livetime for s in spectra])
    rates = np.array([register(s.counts / s.livetime, s.calibration, energy) for s in spectra])
    return ((times / times.sum())[:, None] * rates).sum(axis=0)


def cluster_templates(values, families, labels, threshold):
    """Keep the first representative satisfying both within-family similarities."""
    vectors = values.T
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths == 0):
        raise ValueError("A template has no response in the analysis range.")
    unit = vectors / lengths[:, None]
    cosine = np.clip(unit @ unit.T, -1, 1)
    pearson = np.corrcoef(vectors) if len(vectors) > 1 else np.ones((1, 1))
    representatives, membership = [], []
    for i, family in enumerate(families):
        matches = [r for r in representatives if families[r] == family
                   and cosine[i, r] >= threshold and pearson[i, r] >= threshold]
        representative = matches[0] if matches else i
        if not matches:
            representatives.append(i)
        membership.append(dict(template=labels[i], family=family,
            representative=labels[representative], is_representative=(i == representative),
            cosine=cosine[i, representative], pearson=pearson[i, representative]))
    return np.array(representatives), pd.DataFrame(membership), cosine, pearson


# The background-only count benchmark calibrates the slab, not a hard isotope
# detection limit. The detection errors and the slab quantile are distinct.
# Count-limit equations follow MARLAP (2004), Chapter 20, Eqs. 20.26–20.28.
def detection_limits(backgrounds, sample_time, settings):
    counts = sum(s.counts[(s.energy >= settings.energy_min) &
                          (s.energy <= settings.energy_max)].sum() for s in backgrounds)
    background_time = sum(s.livetime for s in backgrounds)
    expected = counts * sample_time / background_time
    variance = expected * (1 + sample_time / background_time)
    za, zb = norm.ppf(1 - settings.detection_alpha), norm.ppf(1 - settings.detection_beta)
    lc = za * np.sqrt(variance)
    ld = lc + zb**2 / 2 + zb * np.sqrt(variance + lc + zb**2 / 4)
    return dict(background_counts=float(counts), background_livetime_s=background_time,
        expected_background_counts=expected, null_variance=variance,
        LC_counts=lc, LD_counts=ld, LQ_counts=50 + np.sqrt(2500 + 100 * variance))


# Retain the full grid for shifted interpolation; only selected channels enter
# the likelihood. Caching reuses exact interpolated arrays without approximation.
class ShiftedTemplate:
    def __init__(self, grid, values, energy):
        self.interpolator = PchipInterpolator(grid, values, extrapolate=True)
        self.energy = energy
        self.cache = []

    def __call__(self, shift):
        for previous, values in self.cache:
            if previous == shift:
                return values
        values = self.interpolator(self.energy + shift)
        self.cache.insert(0, (float(shift), values))
        self.cache = self.cache[:2]
        return values


@dataclass
class Model:
    spectrum: Spectrum
    grid: np.ndarray
    mask: np.ndarray
    templates: np.ndarray
    backgrounds: np.ndarray
    labels: np.ndarray
    background_labels: np.ndarray
    groups: list
    beta: np.ndarray

    def __post_init__(self):
        self.energy = self.grid[self.mask]
        self.y = self.spectrum.counts[self.mask]
        self.isotope_functions = [ShiftedTemplate(self.grid, x, self.energy) for x in self.templates.T]
        self.background_functions = [ShiftedTemplate(self.grid, b, self.energy) for b in self.backgrounds.T]

    def expected_counts(self, state):
        signal = np.zeros_like(self.energy)
        for j, columns in enumerate(self.groups):
            if state.Z[j]:
                response = sum(state.weights[k] * self.isotope_functions[k](state.theta[j]) for k in columns)
                signal += state.A[j] * response
        background = sum(state.Gamma[b] * f(state.theta[len(ISOTOPES) + b])
                         for b, f in enumerate(self.background_functions))
        return np.maximum(self.spectrum.livetime * (signal + background), 1e-12)

    def loglik(self, state):
        mu = self.expected_counts(state)
        return float(np.sum(self.y * np.log(mu) - mu))


def prepare_model(dataset, filename, settings, output):
    spectrum = dataset.spectra[filename]
    grid = spectrum.energy
    mask = (grid >= settings.energy_min) & (grid <= settings.energy_max)
    families, labels, columns = [], [], []
    for isotope, label, runs in dataset.templates:
        times = np.array([s.livetime for s, _ in runs])
        values = np.array([register(response, s.calibration, grid) for s, response in runs])
        columns.append(((times / times.sum())[:, None] * values).sum(axis=0))
        families.append(isotope)
        labels.append(label)
    candidates = np.column_stack(columns)
    background_candidates = np.column_stack([pooled_background(runs, grid) for runs in dataset.backgrounds.values()])
    background_labels = [f"Background_{i}" for i in range(len(dataset.backgrounds))]
    selections = []
    for prefix, values, family, names, threshold in [
        ("template", candidates, families, labels, settings.isotope_similarity),
        ("background", background_candidates, ["background"] * len(background_labels),
         background_labels, settings.background_similarity),
    ]:
        selected, membership, cosine, pearson = cluster_templates(values[mask], family, names, threshold)
        selections.append(selected)
        membership.to_csv(output / f"{prefix}_clusters.csv", index=False)
        pd.DataFrame(cosine, index=names, columns=names).to_csv(output / f"{prefix}_cosine_similarity.csv")
        pd.DataFrame(pearson, index=names, columns=names).to_csv(output / f"{prefix}_pearson_correlation.csv")
    selected, selected_bg = selections
    templates, backgrounds = candidates[:, selected], background_candidates[:, selected_bg]
    retained_families = np.array(families)[selected]
    groups = [np.flatnonzero(retained_families == name) for name in ISOTOPES]
    limits = detection_limits(dataset.backgrounds[settings.correction_background], spectrum.livetime, settings)
    response = np.array([spectrum.livetime * templates[mask][:, group].sum(axis=0).mean() for group in groups])
    aD = limits["LD_counts"] / response
    beta = gamma.ppf(1 - settings.slab_quantile, a=settings.slab_shape) * aD
    pd.DataFrame(dict(isotope=ISOTOPES, response_counts_per_Bq=response,
        reference_activity_Bq=aD, slab_scale_Bq=beta, slab_mode_Bq=beta / (settings.slab_shape + 1),
        slab_mean_Bq=beta / (settings.slab_shape - 1))).to_csv(output / "slab_priors.csv", index=False)
    (output / "detection_limits.json").write_text(json.dumps(limits, indent=2) + "\n")
    return Model(spectrum, grid, mask, templates, backgrounds, np.array(labels)[selected],
                 np.array(background_labels)[selected_bg], groups, beta)


# A nonnegative spectral fit supplies activity starting values without using
# known mixture composition or previous MCMC files. All isotopes initially
# remain available; inclusion indicators subsequently follow the posterior.
@dataclass
class State:
    A: np.ndarray
    weights: np.ndarray
    Gamma: np.ndarray
    theta: np.ndarray
    Z: np.ndarray

    def copy(self):
        return State(self.A.copy(), self.weights.copy(), self.Gamma.copy(), self.theta.copy(), self.Z.copy())


def initial_state(model, settings):
    k, b = len(ISOTOPES), len(model.background_labels)
    weights = np.zeros(len(model.labels))
    for group in model.groups:
        weights[group] = 1 / len(group)
    shapes = np.column_stack([model.templates[model.mask][:, group].mean(axis=1) for group in model.groups]
                             + [column for column in model.backgrounds[model.mask].T])
    design = model.spectrum.livetime * shapes
    totals = design.sum(axis=0)
    noise = np.sqrt(np.maximum(model.y, 1))
    fitted_counts, _ = nnls(design / totals / noise[:, None], model.y / noise)
    coefficients = fitted_counts / totals
    # A prior-mode starting value keeps zero NNLS coefficients inside the slab;
    # it is not a floor on sampled activities or an isotope selection rule.
    activities = np.maximum(coefficients[:k], model.beta / (settings.slab_shape + 1))
    background = np.maximum(coefficients[k:], 1e-6)
    return State(activities, weights, background, np.zeros(k + b), np.ones(k, dtype=int))


# Metropolis–Hastings updates target the unchanged Poisson spike-and-slab
# posterior. Log-normal and Dirichlet proposals include their Hastings ratios.
def log_dirichlet(values, alpha):
    if np.any(values <= 0):
        return -np.inf
    return gammaln(alpha.sum()) - gammaln(alpha).sum() + np.sum((alpha - 1) * np.log(values))


class Sampler:
    def __init__(self, model, state, settings, rng):
        self.model, self.state, self.settings, self.rng = model, state, settings, rng
        self.ll = model.loglik(state)

    def accept(self, proposal, prior_ratio=0.0, hastings=0.0):
        proposed_ll = self.model.loglik(proposal)
        if np.log(self.rng.uniform()) < proposed_ll - self.ll + prior_ratio + hastings:
            self.state, self.ll = proposal, proposed_ll
            return True
        return False

    def flip(self, j):
        proposal = self.state.copy()
        proposal.Z[j] = 1 - proposal.Z[j]
        proposal.A[j] = (1 / self.rng.gamma(self.settings.slab_shape, 1 / self.model.beta[j])
                         if proposal.Z[j] else 0.0)
        pi = self.settings.inclusion_probability
        ratio = (proposal.Z[j] - self.state.Z[j]) * np.log(pi / (1 - pi))
        # Drawing the birth activity from its slab cancels prior/proposal terms.
        return self.accept(proposal, ratio)

    def activity(self, j, step):
        proposal = self.state.copy()
        old = proposal.A[j]
        new = old * np.exp(step * self.rng.normal())
        proposal.A[j] = new
        ratio = -(self.settings.slab_shape + 1) * np.log(new / old) - self.model.beta[j] * (1 / new - 1 / old)
        return self.accept(proposal, ratio, np.log(new / old))

    def weights(self, j, concentration):
        columns = self.model.groups[j]
        if not self.state.Z[j]:
            self.state.weights[columns] = self.rng.dirichlet(np.ones(len(columns)))
            return False
        proposal = self.state.copy()
        old = self.state.weights[columns]
        new = self.rng.dirichlet(concentration * old + 1)
        proposal.weights[columns] = new
        hastings = log_dirichlet(old, concentration * new + 1) - log_dirichlet(new, concentration * old + 1)
        return self.accept(proposal, hastings=hastings)  # Dirichlet(1) prior is uniform.

    def background(self, b, step):
        proposal = self.state.copy()
        old = proposal.Gamma[b]
        new = old * np.exp(step * self.rng.normal())
        proposal.Gamma[b] = new
        ratio = -(new**2 - old**2) / (2 * self.settings.background_prior_sd**2)
        return self.accept(proposal, ratio, np.log(new / old))

    def shift(self, p, step):
        if p < len(ISOTOPES) and not self.state.Z[p]:
            self.state.theta[p] = self.rng.normal(0, self.settings.shift_prior_sd)
            return False, False
        broad = p >= len(ISOTOPES) and self.rng.uniform() < self.settings.broad_shift_probability
        sd = self.settings.broad_shift_sd if broad else step
        proposal = self.state.copy()
        old = proposal.theta[p]
        new = old + self.rng.normal(0, sd)
        proposal.theta[p] = new
        ratio = -(new**2 - old**2) / (2 * self.settings.shift_prior_sd**2)
        # A state-independent mixture of symmetric random walks is symmetric.
        return self.accept(proposal, ratio), broad


# Chains use distinct seeds and evenly spaced quantiles of the shift prior as
# background starts. Per-parameter proposal adaptation stops at burn-in.
def run_chain(model, base_state, settings, chain):
    rng = np.random.default_rng(settings.seed + 1000 * chain)
    state = base_state.copy()
    k, b = len(ISOTOPES), len(state.Gamma)
    if chain:
        state.A *= np.exp(rng.normal(0, 0.01, k))
        state.Gamma *= np.exp(rng.normal(0, 0.01, b))
        state.theta[:k] += rng.normal(0, 0.002, k)
        for group in model.groups:
            if len(group) > 1:
                state.weights[group] = rng.dirichlet(20000 * state.weights[group] + 1)
    start_shift = settings.shift_prior_sd * norm.ppf((chain + 0.5) / settings.chains)
    state.theta[k:] = start_shift
    sampler = Sampler(model, state, settings, rng)
    steps = dict(A=np.full(k, settings.log_activity_step), weights=np.full(k, settings.weight_concentration),
        Gamma=np.full(b, settings.log_background_step),
        theta=np.r_[np.full(k, settings.isotope_shift_step), np.full(b, settings.background_shift_step)])
    warm = {name: np.zeros((2, len(step))) for name, step in steps.items()}
    retained = {name: np.zeros((2, len(step))) for name, step in steps.items()}
    count = settings.iterations - settings.burnin
    samples = {name: np.zeros((count, len(getattr(state, name))), dtype=getattr(state, name).dtype)
               for name in ("A", "weights", "Gamma", "theta", "Z")}
    samples["loglik"] = np.zeros(count)
    background_trace = np.zeros((settings.iterations, b))
    broad_counts = np.zeros((2, 2), dtype=int)  # local/broad × attempts/acceptances
    flip_counts = np.zeros(k, dtype=int)
    for iteration in range(settings.iterations):
        counters = warm if iteration < settings.burnin else retained
        for j in range(k):
            accepted = sampler.flip(j)
            if iteration >= settings.burnin:
                flip_counts[j] += accepted
        for j in range(k):
            if sampler.state.Z[j]:
                accepted = sampler.activity(j, steps["A"][j])
                counters["A"][:, j] += (1, accepted)
        for j, group in enumerate(model.groups):
            if len(group) > 1:
                active = sampler.state.Z[j]
                accepted = sampler.weights(j, steps["weights"][j])
                if active:
                    counters["weights"][:, j] += (1, accepted)
        for background in range(b):
            accepted = sampler.background(background, steps["Gamma"][background])
            counters["Gamma"][:, background] += (1, accepted)
        for p in range(k + b):
            attempted = p >= k or sampler.state.Z[p]
            accepted, broad = sampler.shift(p, steps["theta"][p])
            if attempted:
                counters["theta"][:, p] += (1, accepted)
            if p >= k and iteration >= settings.burnin:
                broad_counts[int(broad)] += (1, accepted)
        if iteration < settings.burnin and (iteration + 1) % settings.adapt_interval == 0:
            gain = 0.5 / np.sqrt((iteration + 1) / settings.adapt_interval)
            for name, step in steps.items():
                attempts, acceptances = warm[name]
                active = attempts > 0
                direction = -1 if name == "weights" else 1
                step[active] *= np.exp(direction * gain * (acceptances[active] / attempts[active] - settings.target_acceptance))
                low, high = (50, 1e6) if name == "weights" else (1e-5, 0.1 if name == "theta" else 0.2)
                steps[name] = np.clip(step, low, high)
                warm[name].fill(0)
        background_trace[iteration] = sampler.state.theta[k:]
        if iteration >= settings.burnin:
            draw = iteration - settings.burnin
            for name in ("A", "weights", "Gamma", "theta", "Z"):
                samples[name][draw] = getattr(sampler.state, name)
            samples["loglik"][draw] = sampler.ll
        if (iteration + 1) % 5000 == 0:
            print(f"  Chain {chain + 1}: {iteration + 1}/{settings.iterations}", flush=True)
    return dict(samples=samples, background_shift_full=background_trace, start_shift=start_shift,
        steps=steps, acceptance_counts=retained, background_proposal_counts=broad_counts,
        inclusion_flips=flip_counts)


# Save posterior draws and chain diagnostics without interpreting [0,0] as
# physical absence. Constant indicators have no meaningful R-hat or ESS.
def summarize(values):
    import arviz as az
    flat = values.ravel()
    row = dict(mean=float(flat.mean()), sd=float(flat.std(ddof=1)),
        CI95_low=float(np.quantile(flat, .025)), CI95_high=float(np.quantile(flat, .975)),
        Rhat=np.nan, ESS_bulk=np.nan, ESS_tail=np.nan)
    if np.ptp(flat) > 0:
        row.update(Rhat=float(az.rhat(values)), ESS_bulk=float(az.ess(values, method="bulk")),
                   ESS_tail=float(az.ess(values, method="tail", prob=(.05, .95))))
    return row


def save_results(model, results, settings, output):
    arrays = {name: np.stack([r["samples"][name] for r in results]) for name in results[0]["samples"]}
    for name, values in arrays.items():
        np.save(output / f"chains_{name}.npy", values)
    alignment = np.r_[np.array(ISOTOPES), model.background_labels]
    for name, values in dict(isotope_labels=ISOTOPES, template_labels=model.labels,
        alignment_labels=alignment, energy_keV=model.energy, observed_counts=model.y,
        interpolation_energy_keV=model.grid, interpolation_isotope_templates=model.templates,
        interpolation_background_templates=model.backgrounds,
        background_shift_full=np.stack([r["background_shift_full"] for r in results])).items():
        np.save(output / f"{name}.npy", values)
    rows, activities = [], []
    for name, labels in [("A", ISOTOPES), ("Z", ISOTOPES), ("weights", model.labels),
                         ("Gamma", model.background_labels), ("theta", alignment)]:
        for j, label in enumerate(labels):
            stats = summarize(arrays[name][:, :, j])
            rows.append(dict(parameter=f"{name}_{label}", **stats))
            if name == "A":
                z = arrays["Z"][:, :, j]
                activities.append(dict(isotope=label, PIP=float(z.mean()),
                    included_draws=int(z.sum()),
                    accepted_inclusion_flips=int(sum(r["inclusion_flips"][j] for r in results)), **stats,
                    **{f"PIP_chain_{c+1}": float(z[c].mean()) for c in range(settings.chains)}))
    rows.append(dict(parameter="loglik", **summarize(arrays["loglik"])))
    diagnostics = pd.DataFrame(rows)
    diagnostics["flag"] = (diagnostics.Rhat >= 1.01) | (diagnostics.ESS_bulk < 400)
    # A nonconstant parameter with undefined diagnostics also needs attention.
    diagnostics["flag"] |= (diagnostics.sd > 0) & (diagnostics.Rhat.isna() | diagnostics.ESS_bulk.isna())
    diagnostics.to_csv(output / "parameter_diagnostics.csv", index=False)
    activities = pd.DataFrame(activities)
    activities.to_csv(output / "activity_summary.csv", index=False)
    tuning = []
    for c, result in enumerate(results):
        for block, labels in [("A", ISOTOPES), ("weights", ISOTOPES),
                              ("Gamma", model.background_labels), ("theta", alignment)]:
            for j, label in enumerate(labels):
                attempts, acceptances = result["acceptance_counts"][block][:, j]
                tuning.append(dict(chain=c+1, parameter=f"{block}_{label}",
                    final_proposal_scale=result["steps"][block][j], attempts=int(attempts),
                    acceptances=int(acceptances), acceptance_rate=acceptances/attempts if attempts else np.nan))
    pd.DataFrame(tuning).to_csv(output / "proposal_diagnostics.csv", index=False)
    pd.DataFrame([dict(chain=c+1, proposal=label, attempts=int(counts[0]), acceptances=int(counts[1]))
        for c, result in enumerate(results)
        for label, counts in zip(("local", "broad"), result["background_proposal_counts"])
    ]).to_csv(output / "background_proposal_diagnostics.csv", index=False)
    flagged = diagnostics.loc[diagnostics.flag, "parameter"].tolist()
    print(activities[["isotope", "PIP", "mean", "CI95_low", "CI95_high"]].to_string(index=False))
    print("Diagnostics:", ", ".join(flagged) if flagged else "R-hat < 1.01 and bulk ESS >= 400 for nonconstant parameters")
    if flagged:
        print("Pooled summaries are provisional until the flagged chain diagnostics are resolved.")
    return arrays, flagged


# Compare activities with the documented G8 references after decay correction.
# Alternative subset determinations are selected using the source-table notes,
# never summed. E_n uses the 95% interval half-width and reference k=2 uncertainty.
def compare_references(dataset, model, arrays, output):
    table = pd.read_csv(dataset.directory / "021_A_REF_main.csv")
    rows = table.loc[table.MeasurementSeriesID_613 == model.spectrum.series].copy()
    g8 = rows.loc[rows.REF_type == "PeakAnalysis_G8"].copy()
    g8["Subset"] = g8.Subset.fillna("")
    notes = rows.Note.fillna("")
    excluded = [subset for subset in g8.Subset.unique() if subset and
                notes.str.contains(f"{subset} is not included", case=False, regex=False).any()]
    g8 = g8.loc[~g8.Subset.isin(excluded)]
    if g8.empty or g8.Nuclide.duplicated().any():
        raise ValueError(f"No unique G8 reference selection for {model.spectrum.series}.")
    half_lives = pd.read_csv(dataset.directory / "05_Half_lives.csv").set_index("Nuclide").HalfLife_s
    comparisons = []
    for _, row in g8.iterrows():
        reference_date = pd.Timestamp(row.t_ref)
        seconds = (reference_date - model.spectrum.start).total_seconds()
        factor = np.exp(-np.log(2) * seconds / half_lives[row.Nuclide])
        values = arrays["A"][:, :, ISOTOPES.index(row.Nuclide)] * factor
        mean = float(values.mean())
        low, high = np.quantile(values, [.025, .975])
        comparisons.append(dict(isotope=row.Nuclide, reference_date=str(reference_date),
            reference_Bq=row.A_Bq, reference_standard_uncertainty_Bq=row.uA_Bq,
            subset=row.Subset, reference_file=row.REF_file, decay_factor=factor,
            mean_at_reference_Bq=mean, CI95_low_Bq=low, CI95_high_Bq=high,
            bias_percent=100 * (mean / row.A_Bq - 1),
            En=(mean - row.A_Bq) / np.sqrt(((high - low) / 2)**2 + (2 * row.uA_Bq)**2)))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "reference_comparison_G8.csv", index=False)
    print(comparison[["isotope", "bias_percent", "En"]].to_string(index=False))


# Plot fitted counts and retained-chain traces. The mean fitted spectrum uses
# up to 100 evenly spaced draws per chain; activity summaries use every draw.
def plot_results(model, arrays, results, settings, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means, backgrounds = [], []
    for chain in range(settings.chains):
        draws = np.linspace(0, arrays["A"].shape[1]-1, min(100, arrays["A"].shape[1]), dtype=int)
        for draw in draws:
            state = State(*(arrays[key][chain, draw] for key in ("A", "weights", "Gamma", "theta", "Z")))
            means.append(model.expected_counts(state))
            backgrounds.append(model.spectrum.livetime * sum(state.Gamma[b] * f(state.theta[len(ISOTOPES)+b])
                               for b, f in enumerate(model.background_functions)))
    mu, bg = np.mean(means, axis=0), np.mean(backgrounds, axis=0)
    np.savez(output / "spectral_components.npz", energy_keV=model.energy, observed_counts=model.y,
             posterior_mean_counts=mu, posterior_mean_background_counts=bg)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(model.energy, model.y, color=".6", lw=.5, label="Observed")
    axes[0].plot(model.energy, mu, lw=.7, label="Posterior mean fit")
    axes[0].plot(model.energy, bg, lw=.7, label="Background")
    axes[0].set(yscale="log", ylabel="Counts / channel", title=model.spectrum.filename)
    axes[0].legend()
    axes[1].plot(model.energy, (model.y-mu)/np.sqrt(mu), color=".4", lw=.5)
    axes[1].axhline(0, color="k", lw=.5)
    axes[1].set(xlabel="Energy (keV)", ylabel="Pearson residual", xlim=(settings.energy_min, settings.energy_max))
    fig.tight_layout(); fig.savefig(output / "spectrum.png", dpi=160); plt.close(fig)
    active = np.flatnonzero(arrays["Z"].mean(axis=(0, 1)) >= .5)
    b = len(model.background_labels)
    fig, axes = plt.subplots(len(active) + 2*b + 1, 1, figsize=(11, 2*(len(active)+2*b+1)))
    traces = [(arrays["A"][:, :, j], f"{ISOTOPES[j]} activity (Bq)") for j in active]
    traces += [(arrays["Gamma"][:, :, j], f"{label} scale") for j, label in enumerate(model.background_labels)]
    traces += [(np.stack([r["background_shift_full"][:, j] for r in results]), f"{label} shift (keV)")
               for j, label in enumerate(model.background_labels)]
    traces += [(arrays["loglik"], "Log likelihood")]
    for ax, (values, label) in zip(axes, traces):
        full = values.shape[1] == settings.iterations
        offset = 0 if full else settings.burnin
        for chain in range(settings.chains):
            ax.plot(np.arange(values.shape[1])+offset+1, values[chain], lw=.4, alpha=.7, label=f"Chain {chain+1}")
        if full:
            ax.axvline(settings.burnin, color="k", ls="--", lw=.8)
        ax.set_ylabel(label)
    axes[0].legend(ncol=settings.chains); axes[-1].set_xlabel("Iteration")
    fig.tight_layout(); fig.savefig(output / "chain_traces.png", dpi=140); plt.close(fig)


# Select one or all mixtures/runs and write each analysis to its own directory.
# The command-line interface changes data selection and run length only.
def main():
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, help="Default: data directory/results_updated/<timestamp>")
    parser.add_argument("--mixture", default="849-23-P058-111", help="Measurement series ID or 'all'")
    parser.add_argument("--run", default="1", help="One-based run number or 'all'")
    parser.add_argument("--chains", type=int, default=defaults.chains)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--burnin", type=int, default=defaults.burnin)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.chains < 2 or args.jobs < 1 or not 0 <= args.burnin < args.iterations - 3:
        parser.error("Use at least two chains, positive jobs, and at least four retained iterations.")
    settings = Settings(chains=args.chains, iterations=args.iterations, burnin=args.burnin, seed=args.seed)
    output_root = args.output_dir or args.data_dir / "results_updated" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    dataset = Dataset(args.data_dir, settings)
    series_ids = list(dataset.mixtures) if args.mixture == "all" else [args.mixture]
    selections = []
    for series in series_ids:
        if series not in dataset.mixtures:
            parser.error(f"Unknown mixture {series}; choose from {', '.join(dataset.mixtures)}")
        filenames = dataset.mixtures[series]
        try:
            runs = list(range(1, len(filenames)+1)) if args.run == "all" else [int(args.run)]
        except ValueError:
            parser.error("--run must be a positive integer or 'all'.")
        if any(run < 1 or run > len(filenames) for run in runs):
            parser.error(f"{series} has {len(filenames)} runs, numbered from 1.")
        selections += [(series, run, filenames[run-1]) for run in runs]
    for series, run, filename in selections:
        output = output_root / f"{series}_run{run}"
        output.mkdir()  # Existing results are never silently overwritten.
        start = time.perf_counter()
        print(f"\n{series}, run {run}: {filename}", flush=True)
        model = prepare_model(dataset, filename, settings, output)
        state = initial_state(model, settings)
        np.savez(output / "initial_state.npz", **asdict(state))
        if args.jobs == 1:
            results = [run_chain(model, state, settings, c) for c in range(settings.chains)]
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, settings.chains),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(run_chain, model, state, settings, c) for c in range(settings.chains)]
                results = [future.result() for future in futures]
        arrays, flags = save_results(model, results, settings, output)
        compare_references(dataset, model, arrays, output)
        plot_results(model, arrays, results, settings, output)
        metadata = dict(settings=asdict(settings), measurement_series=series, run=run, filename=filename,
            livetime_s=model.spectrum.livetime, background_series=list(dataset.backgrounds),
            shift_starts_keV=[r["start_shift"] for r in results],
            initialization="Nonnegative weighted spectral fit with equal template weights; all inclusion indicators start on",
            diagnostic_flags=flags, wall_seconds=time.perf_counter()-start,
            interpretation="Inspect chain diagnostics. [0,0] does not establish absence; L_D is a background-only count benchmark, not an isotope-specific detection guarantee.")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {output} ({metadata['wall_seconds']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
