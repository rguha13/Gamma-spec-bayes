"""Configuration objects for the Bayesian FSA workflow."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_ISOTOPES = np.array(
    [
        "Na22",
        "Mn54",
        "Co57",
        "Co60",
        "Zn65",
        "Y88",
        "Cd109",
        "Cs134",
        "Cs137",
        "Pb210",
        "Am241",
    ],
    dtype=str,
)


@dataclass(frozen=True)
class PriorConfig:
    """Prior hyperparameters used by the Bayesian FSA model."""

    alpha_ig: float = 2.5
    beta_ig: float = 1500.0
    inclusion_probability: float = 0.5
    sigma_theta_channels: float = 0.05
    gamma_scale: float = 0.5


@dataclass(frozen=True)
class ProposalConfig:
    """Random-walk proposal scales for continuous Metropolis updates."""

    step_log_activity: float = 0.05
    step_gamma: float = 0.01
    step_theta: float = 0.03


@dataclass(frozen=True)
class MCMCConfig:
    """MCMC run settings."""

    n_chains: int = 4
    n_iter: int = 20000
    burnin: int = 5000
    base_seed: int = 123
    pip_threshold: float = 0.5

    def validate(self) -> None:
        if self.n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if not (0 <= self.burnin < self.n_iter):
            raise ValueError("burnin must satisfy 0 <= burnin < n_iter.")
        if not (0.0 <= self.pip_threshold <= 1.0):
            raise ValueError("pip_threshold must be between 0 and 1.")


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level analysis configuration."""

    data_dir: Path
    output_dir: Path
    mixture_index: int = 1
    run_index: int = 0
    isotopes: np.ndarray = field(default_factory=lambda: DEFAULT_ISOTOPES.copy())

    def result_dir_name(self, measurement_series_id: str) -> str:
        return f"{measurement_series_id}_run{self.run_index}"
