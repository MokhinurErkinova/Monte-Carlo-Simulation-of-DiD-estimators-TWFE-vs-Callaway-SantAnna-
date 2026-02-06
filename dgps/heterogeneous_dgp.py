"""
dgps/heterogeneous_dgp.py — Stage 2: Heterogeneous effects, same timing.

DGP: Y_it = alpha_i + lambda_t + tau_group * D_it + eps_it

Panel is emitted in (id, t) order.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

class HeterogeneousDGP:
    def __init__(self, group_sizes: Dict[str, int],
                 group_effects: Dict[str, float],
                 T: int = 10, g: int = 6,
                 sigma_alpha: float = 1.0, sigma_eps: float = 1.0):
        self.group_sizes = group_sizes
        self.group_effects = group_effects
        self.T = T
        self.g = g
        self.sigma_alpha = sigma_alpha
        self.sigma_eps = sigma_eps

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        T = self.T

        group_labels = []
        tau_unit = []
        G_unit = []

        for grp_name, n_units in self.group_sizes.items():
            group_labels.append(np.full(n_units, grp_name))
            tau_unit.append(np.full(n_units, self.group_effects.get(grp_name, 0.0)))
            G_unit.append(np.full(n_units, 0 if grp_name == "control" else self.g, dtype=int))

        group_arr = np.concatenate(group_labels)
        tau_arr = np.concatenate(tau_unit)
        G_arr = np.concatenate(G_unit)
        N = len(G_arr)

        alpha = rng.normal(0.0, self.sigma_alpha, size=N)

        id_panel = np.repeat(np.arange(1, N + 1), T)
        t_panel = np.tile(np.arange(1, T + 1), N)
        G_panel = np.repeat(G_arr, T)
        D_panel = ((G_panel > 0) & (t_panel >= G_panel)).astype(int)
        alpha_panel = np.repeat(alpha, T)
        tau_panel = np.repeat(tau_arr, T)
        lambda_panel = t_panel.astype(np.float64)
        eps_panel = rng.normal(0.0, self.sigma_eps, size=N * T)

        Y_panel = alpha_panel + lambda_panel + tau_panel * D_panel + eps_panel

        df = pd.DataFrame({
            "id": id_panel,
            "t": t_panel,
            "group": np.repeat(group_arr, T),
            "G": G_panel,
            "D": D_panel,
            "alpha": alpha_panel,
            "lambda_t": lambda_panel,
            "tau_i": tau_panel,
            "eps": eps_panel,
            "Y": Y_panel,
        })
        return df

    @property
    def true_att(self) -> float:
        treated = {k: v for k, v in self.group_sizes.items() if k != "control"}
        total = sum(treated.values())
        return sum(self.group_sizes[g] * self.group_effects[g] for g in treated) / total
