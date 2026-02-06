"""
dgps/deterministic_dgp.py — Stage 0: Deterministic unit test.

DGP: Y_it = lambda_t + tau * D_it, lambda_t = t

No unit FE, no noise. Both estimators must recover tau exactly.
"""

import numpy as np
import pandas as pd
from typing import Optional

class DeterministicDGP:
    def __init__(self, N_treated: int = 50, N_control: int = 50,
                 T: int = 10, g: int = 6, tau: float = 1.0):
        self.N_treated = N_treated
        self.N_control = N_control
        self.T = T
        self.g = g
        self.tau = tau

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        N = self.N_treated + self.N_control
        ids = np.arange(1, N + 1)
        group = np.array(["treated"] * self.N_treated + ["control"] * self.N_control)

        df = pd.DataFrame({
            "id": np.repeat(ids, self.T),
            "t": np.tile(np.arange(1, self.T + 1), N),
            "group": np.repeat(group, self.T),
        })

        df["G"] = np.where(df["group"] == "treated", self.g, 0)
        df["D"] = ((df["G"] > 0) & (df["t"] >= df["G"])).astype(int)
        df["Y"] = df["t"].astype(float) + self.tau * df["D"]

        return df

    @property
    def true_att(self) -> float:
        return self.tau
