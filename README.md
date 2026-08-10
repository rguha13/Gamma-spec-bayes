# Bayesian Full-Spectrum Analysis for HPGe Gamma Spectra

This repository contains the implementation used for Bayesian radionuclide
identification and activity estimation from high-resolution HPGe gamma-ray
spectra. The analysis uses the experimental dataset of Wübbeler et al.

## Model

For observed channel count `Y_c` and livetime `T`, the expected count is

```text
Y_c ~ Poisson(mu_c)

mu_c = T * [sum_j Z_j A_j sum_m w_jm X_jm(E_c + theta_j)
            + sum_b gamma_b B_b(E_c + phi_b)]
```

where:

- `Z_j` is the inclusion indicator for isotope `j`;
- `A_j` is its activity in Bq;
- `X_jm` is retained representative template `m` for isotope `j`, normalized
  to cps/Bq and registered to the selected mixture spectrum's energy grid;
- `w_jm` is a nonnegative simplex weight over the retained templates of
  isotope `j`;
- `theta_j` is an isotope-specific energy shift in keV;
- `B_b`, `gamma_b`, and `phi_b` are the empirical background template,
  nonnegative scale, and energy shift.

The isotope library contains 31 candidate template series for 11 radionuclides.
Within each isotope, near-duplicate templates are clustered over 20--1500 keV
when both cosine similarity and Pearson correlation are at least 0.999. The
first template in each cluster is retained. For the supplied dataset this
reduces the library from 31 candidates to 18 representatives. The two
background series are screened with the same criteria and reduce to one
representative background component.

The priors used by the final analysis are:

```text
Z_j ~ Bernoulli(0.5)
A_j | Z_j = 1 ~ Inverse-Gamma(alpha=2.5, beta=4500 Bq)
A_j | Z_j = 0 = 0
w_j ~ Dirichlet(1, ..., 1)
theta_j, phi_b ~ Normal(0, 0.5^2 keV^2)
gamma_b ~ Half-Normal(0.5)
```

Posterior inference uses a custom Metropolis-within-Gibbs sampler. The default
run uses four chains of 20,000 iterations with the first 5,000 iterations of
each chain discarded as burn-in.

## Files

- `gamma_spec_bayes.py`: final sampler, clustering, diagnostics, posterior
  summaries, and PeakAnalysis_G8 comparison.
- `plotting.py`: regenerates manuscript figures from saved posterior draws
  without rerunning MCMC.
- `requirements.txt`: Python dependencies.

## Data

Download the Wübbeler et al. dataset separately and place the following files
in one directory:

```text
02_A_REF.xlsx
03_Templates_A_REF.xlsx
042_DATA_ECal.xlsx
04_DATA_spectra.xlsx
021_A_REF_main.csv
05_Half_lives.csv
```

The data are available from the dataset cited below. They are not duplicated in
this repository.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the analysis

The manuscript example uses measurement series `849-23-P058-111`, Python run
index `0` (`849-23-P058-001.cnf`). It is selected by the default mixture and run
indices:

```bash
python gamma_spec_bayes.py --data-dir /path/to/data
```

For a quick execution check:

```bash
python gamma_spec_bayes.py --data-dir /path/to/data \
  --n-chains 2 --n-iter 200 --burnin 100
```

Any available mixture and run can be selected explicitly:

```bash
python gamma_spec_bayes.py --data-dir /path/to/data \
  --mixture-index 1 --run-index 0
```

Numerical outputs and retained posterior chains are written under
`<data-dir>/results` by default. Use `--output-dir` to choose another location.

## Regenerate figures

After the sampler has saved its chains, figures can be regenerated without
rerunning MCMC:

```bash
python plotting.py --data-dir /path/to/data --mixture 1 --run 0
```

Use `--results-dir` if the saved chains are in a non-default location.

## Principal outputs

The output directory contains the retained chains for activities, inclusion
indicators, template weights, background coefficients, and energy shifts;
template and background cluster assignments; convergence diagnostics printed
to the terminal; posterior summaries; the PeakAnalysis_G8 reference comparison;
and the reconstructed-spectrum and posterior figures.

## Citation

```bibtex
@dataset{wubbeler2025fsa,
  author       = {Wübbeler, Gerd and Stein, Markus and Fleischhack, Holger and Röttger, Stefan and Honig, Andreas},
  title        = {Bayesian full-spectrum analysis of high-resolution gamma-ray spectra with energy scale correction and MCMC uncertainty quantification: Dataset and software},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15631392},
  url          = {https://doi.org/10.5281/zenodo.15631392}
}
```
