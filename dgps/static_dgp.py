"""
dgps/static_dgp.py — Stage 1: Homogeneous effects, single timing.

DGP: Y_it = alpha_i + lambda_t + tau * D_it + eps_it

Panel is emitted in (id, t) order: id repeats T times, t cycles 1..T.
"""

import numpy as np
import pandas as pd
from typing import Optional

class StaticNormalDGP:
    def __init__(self, N_treated: int = 50, N_control: int = 50,
                 T: int = 10, g: int = 6, tau: float = 1.0,
                 sigma_alpha: float = 1.0, sigma_eps: float = 1.0):
        self.N_treated = N_treated
        self.N_control = N_control
        self.T = T
        self.g = g
        self.tau = tau
        self.sigma_alpha = sigma_alpha
        self.sigma_eps = sigma_eps

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        N = self.N_treated + self.N_control
        T = self.T

        G_unit = np.concatenate([
            np.full(self.N_treated, self.g, dtype=int),
            np.zeros(self.N_control, dtype=int),
        ])
        alpha = rng.normal(0.0, self.sigma_alpha, size=N)

        id_panel = np.repeat(np.arange(1, N + 1), T)
        t_panel = np.tile(np.arange(1, T + 1), N)
        G_panel = np.repeat(G_unit, T)
        D_panel = ((G_panel > 0) & (t_panel >= G_panel)).astype(int)
        alpha_panel = np.repeat(alpha, T)
        lambda_panel = t_panel.astype(np.float64)
        eps_panel = rng.normal(0.0, self.sigma_eps, size=N * T)

        Y_panel = alpha_panel + lambda_panel + self.tau * D_panel + eps_panel

        df = pd.DataFrame({
            "id": id_panel,
            "t": t_panel,
            "group": np.repeat(
                np.array(["treated"] * self.N_treated
                        + ["control"] * self.N_control), T),
            "G": G_panel,
            "D": D_panel,
            "alpha": alpha_panel,
            "lambda_t": lambda_panel,
            "eps": eps_panel,
            "Y": Y_panel,
        })
        return df

    @property
    def true_att(self) -> float:
        return self.tau
