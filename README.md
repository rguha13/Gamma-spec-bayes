# Bayesian Full-Spectrum Analysis for HPGe Gamma Spectra

This repository contains the implementation used for Bayesian radionuclide
identification and activity estimation from high-resolution HPGe gamma-ray
spectra. The analysis uses the experimental dataset of Wübbeler et al.

## Model

For observed channel count `Y_c` and livetime `T`, the expected count is

```text
Y_c ~ Poisson(mu_c)

mu_c = T * [sum_j Z_j A_j sum_m w_jm X_jm(E_c + theta_j)
            + sum_b gamma_b B_b(E_c + theta_b)]
```

where:

- `Z_j` is the inclusion indicator for isotope `j`;
- `A_j` is its activity in Bq;
- `X_jm` is retained representative template `m` for isotope `j`, normalized
  to cps/Bq and registered to the selected mixture spectrum's energy grid;
- `w_jm` is a nonnegative simplex weight over the retained templates of
  isotope `j`;
- `theta_j` is an isotope-specific energy shift in keV;
- `B_b`, `gamma_b`, and `theta_b` are the empirical background template,
  nonnegative scale, and energy shift.

The isotope library contains 31 candidate template series for 11 radionuclides.
The sampler estimates the nonnegative source response of each calibration run
after subtracting the estimated background count rate, then normalizes by
reference activity. It fits the observed mixture counts over 20–2800 keV.
Within each isotope, templates are clustered using cosine similarity and
Pearson correlation (threshold 0.999 for both). In the manuscript example,
19 isotope templates and one empirical background component are retained.

Each isotope has prior inclusion probability 0.5 and a point mass at zero
when excluded. Given inclusion, its activity follows an inverse-gamma prior
with shape 2.5. The scale is set separately for each isotope so that the
background-based activity benchmark is the 0.5th percentile of the slab.
The benchmark uses the background measurements, sample livetime, and
retained template responses. The sampler uses uniform Dirichlet template
weights, zero-centered normal energy shifts (SD 0.5 keV), and half-normal
background scaling coefficients (scale 0.5).

The default run uses four Metropolis–Hastings chains of 20,000 iterations
each and discards the first 10,000 per chain. The background shift uses both
local and broader proposals, with distinct initial shifts across chains.

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

The manuscript example is run **1** of measurement series
`849-23-P058-111` (`849-23-P058-001.cnf`). Run numbers start at 1.

```bash
python gamma_spec_bayes.py --data-dir /path/to/data --mixture 849-23-P058-111 --run 1
```

For all spectra from all four mixtures:

```bash
python gamma_spec_bayes.py --data-dir /path/to/data --mixture all --run all
```

The sampler writes one directory per spectrum under
`<data-dir>/results_updated/<timestamp>/`. Use `--output-dir` to choose a
different batch directory. Outputs include the retained draws, posterior
summaries, template clusters, detection benchmark, isotope-specific priors,
convergence diagnostics and comparison with the G8 reference activities.

## Regenerate manuscript figures and tables

Supply the representative spectrum's saved directory and the original input
directory to export figures as PDF and PNG and tables as CSV. Replace
`TIMESTAMP` with the directory name printed by the sampler.

```bash
python plotting.py --data-dir /path/to/data --results-dir /path/to/data/results_updated/TIMESTAMP/849-23-P058-111_run1
```

To report aggregate relative bias, normalized error and isotope selection
from a batch, supply its parent directory:

```bash
python plotting.py --aggregate-root /path/to/data/results_updated/TIMESTAMP
```

## Citation

Wübbeler, G., Stein, M., Fleischhack, H., Röttger, S., & Honig, A. (2025). Bayesian full-spectrum analysis of high-resolution gamma-ray spectra with energy scale correction and MCMC uncertainty quantification: Dataset and software [Dataset]. Zenodo. [https://doi.org/10.5281/zenodo.15631392](https://doi.org/10.5281/zenodo.15631392)
