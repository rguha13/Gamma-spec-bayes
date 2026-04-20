"""Posterior summaries, convergence diagnostics, and reference comparisons."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


REFERENCE_TYPES = [
    "PeakAnalysis_singleEval",
    "PeakAnalysis_G8",
    "PeakAnalysis_complete",
    "MassFraction",
]


def save_chains(result, output_dir: Path) -> None:
    """Persist MCMC chains as NumPy arrays."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "chains_A.npy", result.activity)
    np.save(output_dir / "chains_Gamma.npy", result.gamma)
    np.save(output_dir / "chains_theta.npy", result.theta)
    np.save(output_dir / "chains_Z.npy", result.z)
    np.save(output_dir / "chains_loglik.npy", result.loglik)


def pooled_samples(result):
    """Flatten chains into pooled posterior samples."""

    n_chains, n_draws, k_isotopes = result.activity.shape
    _, _, n_background = result.gamma.shape
    return {
        "activity": result.activity.reshape(n_chains * n_draws, k_isotopes),
        "gamma": result.gamma.reshape(n_chains * n_draws, n_background),
        "theta": result.theta.reshape(n_chains * n_draws, k_isotopes),
        "z": result.z.reshape(n_chains * n_draws, k_isotopes),
    }


def posterior_summary_table(result, isotope_names, b_calibration, pip_threshold=0.5) -> pd.DataFrame:
    """Create a compact posterior summary table for isotope parameters."""

    pooled = pooled_samples(result)
    activity = pooled["activity"]
    theta = pooled["theta"]
    z = pooled["z"]
    pip = z.mean(axis=0)

    rows = []
    for j, isotope in enumerate(isotope_names):
        activity_kbq = activity[:, j] / 1e3
        theta_kev = theta[:, j] * b_calibration
        rows.append(
            {
                "Isotope": isotope,
                "PIP": pip[j],
                "Selected": bool(pip[j] >= pip_threshold),
                "Activity_mean_kBq": activity_kbq.mean(),
                "Activity_ci95_lo_kBq": np.percentile(activity_kbq, 2.5),
                "Activity_ci95_hi_kBq": np.percentile(activity_kbq, 97.5),
                "Shift_mean_keV": theta_kev.mean(),
                "Shift_ci95_lo_keV": np.percentile(theta_kev, 2.5),
                "Shift_ci95_hi_keV": np.percentile(theta_kev, 97.5),
            }
        )
    return pd.DataFrame(rows)


def background_summary_table(result) -> pd.DataFrame:
    """Create posterior summaries for background coefficients."""

    gamma = pooled_samples(result)["gamma"]
    rows = []
    for b in range(gamma.shape[1]):
        rows.append(
            {
                "Background": f"b{b}",
                "Gamma_mean": gamma[:, b].mean(),
                "Gamma_ci95_lo": np.percentile(gamma[:, b], 2.5),
                "Gamma_ci95_hi": np.percentile(gamma[:, b], 97.5),
            }
        )
    return pd.DataFrame(rows)


def arviz_diagnostics(result, isotope_names, selected_idx) -> pd.DataFrame:
    """Compute R-hat and ESS for active isotope parameters and backgrounds."""

    n_background = result.gamma.shape[2]
    idata_dict = {
        **{f"A_{isotope_names[j]}": result.activity[:, :, j] for j in range(len(isotope_names))},
        **{f"theta_{isotope_names[j]}": result.theta[:, :, j] for j in range(len(isotope_names))},
        **{f"gamma_{b}": result.gamma[:, :, b] for b in range(n_background)},
    }
    idata = az.from_dict(posterior=idata_dict)
    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")

    rows = []
    for j in selected_idx:
        for prefix in ("A", "theta"):
            name = f"{prefix}_{isotope_names[j]}"
            rows.append(
                {
                    "Parameter": name,
                    "Rhat": float(rhat[name].values),
                    "ESS_bulk": float(ess_bulk[name].values),
                    "ESS_tail": float(ess_tail[name].values),
                }
            )

    for b in range(n_background):
        name = f"gamma_{b}"
        rows.append(
            {
                "Parameter": name,
                "Rhat": float(rhat[name].values),
                "ESS_bulk": float(ess_bulk[name].values),
                "ESS_tail": float(ess_tail[name].values),
            }
        )

    return pd.DataFrame(rows)


def reference_comparison_table(
    result,
    activity_refs: pd.DataFrame,
    run_filename: str,
    isotope_names,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare posterior activities to selected-run single-evaluation references."""

    pooled = pooled_samples(result)
    activity = pooled["activity"]
    pip = pooled["z"].mean(axis=0)
    iso_idx = {name: i for i, name in enumerate(isotope_names)}

    run_refs = activity_refs[activity_refs["Filename"].astype(str) == run_filename].copy()
    reference_isos = [
        isotope for isotope in isotope_names
        if isotope in set(run_refs["Nuclide"].astype(str))
    ]

    single_eval_refs = run_refs[run_refs["REF_type"].astype(str) == "PeakAnalysis_singleEval"]
    missing = [
        isotope for isotope in reference_isos
        if single_eval_refs[single_eval_refs["Nuclide"].astype(str) == isotope].shape[0] != 1
    ]
    if missing:
        raise ValueError(
            "Expected exactly one PeakAnalysis_singleEval reference for each "
            f"selected-run isotope in {run_filename}; check {missing}."
        )

    comparison_rows = []
    for isotope in reference_isos:
        j = iso_idx[isotope]
        samples = activity[:, j]
        post_mean = samples.mean()
        ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])
        u_post = (ci_hi - ci_lo) / 2.0

        ref = single_eval_refs[single_eval_refs["Nuclide"].astype(str) == isotope].iloc[0]
        ref_activity = float(ref["A_Bq"])
        ref_uncertainty = float(ref["uA_Bq"])
        u_ref_95 = 2.0 * ref_uncertainty

        comparison_rows.append(
            {
                "Filename": run_filename,
                "Isotope": isotope,
                "Post_mean_kBq": post_mean / 1e3,
                "CI_lo_kBq": ci_lo / 1e3,
                "CI_hi_kBq": ci_hi / 1e3,
                "PIP": pip[j],
                "Ref_singleEval_kBq": ref_activity / 1e3,
                "uRef_k1_kBq": ref_uncertainty / 1e3,
                "t_ref": ref["t_ref"],
                "Rel_bias_pct": (post_mean - ref_activity) / ref_activity * 100.0,
                "En": (post_mean - ref_activity) / np.sqrt(u_post**2 + u_ref_95**2),
                "Covered_95pct": bool(ci_lo <= ref_activity <= ci_hi),
            }
        )

    all_refs = run_refs.sort_values(["Nuclide", "REF_type"]).copy()
    all_refs = all_refs[["Filename", "Nuclide", "REF_type", "A_Bq", "uA_Bq", "t_ref"]]
    all_refs["A_kBq"] = all_refs["A_Bq"] / 1e3
    all_refs["uA_k1_kBq"] = all_refs["uA_Bq"] / 1e3
    all_refs = all_refs.drop(columns=["A_Bq", "uA_Bq"])

    return pd.DataFrame(comparison_rows), all_refs

