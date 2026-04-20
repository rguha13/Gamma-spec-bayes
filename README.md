# Bayesian Full-Spectrum Analysis for HPGe Gamma Spectra

This repository contains a Python implementation of a Bayesian full-spectrum
analysis (FSA) model for radionuclide identification and activity estimation in
high-resolution HPGe gamma-ray spectra.

The model treats measured channel counts as Poisson random variables whose mean
is a nonnegative mixture of isotope templates and empirical background
templates. Candidate isotope selection is handled with spike-and-slab activity
priors, and small energy-calibration mismatch is handled with per-isotope
template shift parameters.

## Model Summary

For a selected spectrum with livetime `T`, channel counts `Y_c`, candidate
isotopes `j = 1,...,K`, and background templates `b = 1,...,B`, the model is

```text
Y_c ~ Poisson(mu_c)

mu_c = T * [sum_j Z_j A_j X_j(c + theta_j)
            + sum_b gamma_b X_b(c)]
```

where:

- `A_j` is the activity of isotope `j` in Bq.
- `Z_j` is a binary inclusion indicator for isotope `j`.
- `X_j` is the isotope template normalized to cps/Bq.
- `theta_j` is a small channel shift for isotope `j`.
- `gamma_b` is the nonnegative coefficient for background template `b`.
- `X_b` is a background template normalized to cps.

Posterior inference is performed with a custom Metropolis-within-Gibbs sampler.
Posterior inclusion probabilities (PIPs) are used as evidence for isotope
presence.

## Repository Layout

```text
Github_files/
  README.md
  requirements.txt
  run_analysis.py
  bayes_fsa/
    __init__.py
    config.py
    data.py
    templates.py
    model.py
    mcmc.py
    summaries.py
    plots.py
```

## Data Files

The analysis expects the Wubbeler et al. dataset files in one data directory:

```text
02_A_REF.xlsx
03_Templates_A_REF.xlsx
042_DATA_ECal.xlsx
04_DATA_spectra.xlsx
05_Half_lives.csv
```

By default, `run_analysis.py` assumes these files are in the parent directory of
`Github_files`. You can override this with `--data-dir`.

## Installation

Python 3.10 or newer is recommended.

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

To confirm the environment is ready:

```bash
python -c "import numpy, pandas, scipy, matplotlib, arviz, openpyxl; print('OK')"
```

## Basic Usage

From inside `Github_files`, run:

```bash
python run_analysis.py
```

The default configuration analyzes mixture index `1`, run index `0`, which is
the `849-23-P058-111` mixture series with `Am241`, `Cd109`, `Co57`, and
`Pb210`.

To analyze a different run from the same mixture:

```bash
python run_analysis.py --mixture-index 1 --run-index 7
```

To perform a short smoke-test run:

```bash
python run_analysis.py --n-iter 200 --burnin 100 --n-chains 2
```

## Outputs

Results are written to:

```text
<output-dir>/<MeasurementSeriesID>_run<run-index>/
```

The output directory contains:

- `chains_A.npy`
- `chains_Gamma.npy`
- `chains_theta.npy`
- `chains_Z.npy`
- `chains_loglik.npy`
- `posterior_summary.csv`
- `diagnostics.csv`
- `reference_comparison.csv`
- `reference_activity_methods_selected_run.csv`
- diagnostic and posterior figures

Reference comparisons use `PeakAnalysis_singleEval` from `02_A_REF.xlsx` for
the selected run/file. This keeps the comparison on the same single-spectrum
basis as the Bayesian analysis.

## Notes for Reproducibility

- Activities are reported in kBq in output tables, but sampled internally in Bq.
- Shift parameters are sampled in channels and reported in keV using the
  selected run's linear energy calibration.
- `05_Half_lives.csv` is loaded for completeness, but the reference activities
  in `02_A_REF.xlsx` are already decay-corrected to the selected measurement
  start time.
- Random seeds are deterministic by default. Chain `c` uses
  `base_seed + 1000*c`.
