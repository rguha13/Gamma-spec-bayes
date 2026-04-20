"""Run Bayesian full-spectrum analysis for one mixture spectrum."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from bayes_fsa.config import AnalysisConfig, DEFAULT_ISOTOPES, MCMCConfig, PriorConfig, ProposalConfig
from bayes_fsa.data import load_dataset
from bayes_fsa.mcmc import run_multiple_chains
from bayes_fsa.plots import (
    plot_activity_posteriors,
    plot_activity_traces,
    plot_background_posteriors,
    plot_isotope_templates,
    plot_shift_posteriors,
    plot_spectral_reconstruction,
)
from bayes_fsa.summaries import (
    arviz_diagnostics,
    background_summary_table,
    posterior_summary_table,
    reference_comparison_table,
    save_chains,
)
from bayes_fsa.templates import build_template_library


def parse_args():
    default_data_dir = Path(__file__).resolve().parents[1]
    default_output_dir = default_data_dir / "results_github"

    parser = argparse.ArgumentParser(
        description="Bayesian full-spectrum analysis for HPGe gamma spectra."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--mixture-index", type=int, default=1)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--pip-threshold", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true", help="Reduce sampler progress output.")
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = AnalysisConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mixture_index=args.mixture_index,
        run_index=args.run_index,
        isotopes=DEFAULT_ISOTOPES,
    )
    prior = PriorConfig()
    proposal = ProposalConfig()
    mcmc = MCMCConfig(
        n_chains=args.n_chains,
        n_iter=args.n_iter,
        burnin=args.burnin,
        base_seed=args.base_seed,
        pip_threshold=args.pip_threshold,
    )
    mcmc.validate()

    print("Loading dataset...")
    dataset = load_dataset(analysis.data_dir, analysis.isotopes)
    print(f"  Isotope candidates: {len(analysis.isotopes)}")
    print(f"  Background series: {len(dataset.background_spectra)}")
    print(f"  Mixture series: {len(dataset.mixture_spectra)}")

    print("\nBuilding template library...")
    template_library = build_template_library(dataset.single_spectra, dataset.background_spectra)
    print(f"  Isotope template matrix: {template_library.isotope_templates.shape}")
    print(f"  Background template matrix: {template_library.background_templates.shape}")

    mixture = dataset.mixture_spectra[analysis.mixture_index]
    counts = mixture["Counts"][analysis.run_index, :].astype(float)
    livetime = float(mixture["Livetime"][analysis.run_index])
    run_filename = str(mixture["Filename"][analysis.run_index])
    output_dir = analysis.output_dir / analysis.result_dir_name(mixture["MSID"])
    output_dir.mkdir(parents=True, exist_ok=True)

    a_cal, b_cal = [float(x) for x in mixture["Ecal"][analysis.run_index, :]]
    energy = a_cal + b_cal * template_library.channels

    print("\nSelected analysis case")
    print(f"  Mixture index: {analysis.mixture_index}")
    print(f"  Measurement series: {mixture['MSID']}")
    print(f"  Run index: {analysis.run_index}")
    print(f"  Filename: {run_filename}")
    print(f"  Livetime: {livetime:.3f} s")
    print(f"  Output directory: {output_dir}")

    print("\nStarting MCMC")
    print(f"  Chains: {mcmc.n_chains}")
    print(f"  Iterations per chain: {mcmc.n_iter}")
    print(f"  Burn-in: {mcmc.burnin}")
    start = time.time()
    result = run_multiple_chains(
        counts=counts,
        livetime=livetime,
        template_library=template_library,
        prior=prior,
        proposal=proposal,
        mcmc=mcmc,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start
    print(f"\nMCMC complete in {elapsed / 60.0:.2f} minutes")

    print("\nSaving chains and summaries...")
    save_chains(result, output_dir)

    summary = posterior_summary_table(
        result,
        isotope_names=analysis.isotopes,
        b_calibration=b_cal,
        pip_threshold=mcmc.pip_threshold,
    )
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)

    selected_idx = np.where(summary["PIP"].to_numpy() >= mcmc.pip_threshold)[0]
    diagnostics = arviz_diagnostics(result, analysis.isotopes, selected_idx)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)

    background_summary = background_summary_table(result)
    background_summary.to_csv(output_dir / "background_summary.csv", index=False)

    reference_comparison, all_reference_methods = reference_comparison_table(
        result=result,
        activity_refs=dataset.activity_refs,
        run_filename=run_filename,
        isotope_names=analysis.isotopes,
    )
    reference_comparison.to_csv(output_dir / "reference_comparison.csv", index=False)
    all_reference_methods.to_csv(output_dir / "reference_activity_methods_selected_run.csv", index=False)

    print("\nPosterior inclusion probabilities")
    for _, row in summary.iterrows():
        print(f"  {row['Isotope']:6s}: {row['PIP']:.4f}")

    print("\nReference comparison: PeakAnalysis_singleEval")
    if not reference_comparison.empty:
        print(reference_comparison.to_string(index=False))

    print("\nWriting figures...")
    plot_spectral_reconstruction(counts, livetime, result, template_library, energy, output_dir)
    plot_activity_traces(result, analysis.isotopes, selected_idx, output_dir)
    plot_activity_posteriors(result, analysis.isotopes, selected_idx, output_dir)
    plot_shift_posteriors(result, analysis.isotopes, selected_idx, b_cal, output_dir)
    plot_background_posteriors(result, output_dir)
    plot_isotope_templates(template_library, analysis.isotopes, energy, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
