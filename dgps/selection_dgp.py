"""
dgps/selection_dgp.py — Stage 3B: Selection into treatment timing.

Two DGPs:
SelectionTimingDGP — selection on b_i (treatment gains). PT holds.
SelectionPretrendsDGP — selection on b_i BUT with group-specific trends. PT violated.

Panel is emitted in (id, t) order.
"""

import numpy as np
import pandas as pd
from typing import Optional

class SelectionTimingDGP:
    """
    Y_it = alpha_i + lambda_t + b_i * D_it + eps_it

    Units sorted by b_i: top share_early → G=g_early,
    next share_late → G=g_late, bottom → G=0 (never).

    Parallel trends holds.
    """

    def __init__(self, N: int = 100, T: int = 10,
                 g_early: int = 4, g_late: int = 7,
                 share_never: float = 0.40,
                 share_early: float = 0.30,
                 share_late: float = 0.30,
                 mu_b: float = 1.0, sigma_b: float = 0.5,
                 sigma_alpha: float = 1.0, sigma_eps: float = 1.0,
                 b_fixed: Optional[str] = None):
        self.N = N; self.T = T
        self.g_early = g_early; self.g_late = g_late
        self.share_never = share_never
        self.share_early = share_early
        self.share_late = share_late
        self.mu_b = mu_b; self.sigma_b = sigma_b
        self.sigma_alpha = sigma_alpha
        self.sigma_eps = sigma_eps

        if b_fixed == "use_fixed":
            rng_init = np.random.default_rng(99999)
            self._b_fixed = rng_init.normal(mu_b, sigma_b, size=N)
        else:
            self._b_fixed = None

        self._true_att = mu_b

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        N, T = self.N, self.T

        b = self._b_fixed.copy() if self._b_fixed is not None else \
            rng.normal(self.mu_b, self.sigma_b, size=N)

        q_never = np.quantile(b, self.share_never)
        q_late = np.quantile(b, self.share_never + self.share_late)

        G_unit = np.zeros(N, dtype=int)
        G_unit[b > q_late] = self.g_early
        G_unit[(b > q_never) & (b <= q_late)] = self.g_late

        treated_mask = G_unit > 0
        N_early = int((G_unit == self.g_early).sum())
        N_late = int((G_unit == self.g_late).sum())
        self._true_att = float(
            (N_early * b[G_unit == self.g_early].mean() +
             N_late * b[G_unit == self.g_late].mean()) / (N_early + N_late)
        )

        alpha = rng.normal(0.0, self.sigma_alpha, size=N)

        id_panel = np.repeat(np.arange(1, N + 1), T)
        t_panel = np.tile(np.arange(1, T + 1), N)
        G_panel = np.repeat(G_unit, T)
        D_panel = ((G_panel > 0) & (t_panel >= G_panel)).astype(int)
        alpha_panel = np.repeat(alpha, T)
        b_panel = np.repeat(b, T)
        lambda_panel = t_panel.astype(np.float64)
        eps_panel = rng.normal(0.0, self.sigma_eps, size=N * T)

        Y_panel = alpha_panel + lambda_panel + b_panel * D_panel + eps_panel

        df = pd.DataFrame({
            "id": id_panel,
            "t": t_panel,
            "G": G_panel,
            "D": D_panel,
            "alpha": alpha_panel,
            "lambda_t": lambda_panel,
            "b_i": b_panel,
            "eps": eps_panel,
            "Y": Y_panel,
        })
        return df

    @property
    def true_att(self) -> float:
        return self._true_att


class SelectionPretrendsDGP:
    """
    Y_it = alpha_i + lambda_t + theta_g * t + b_i * D_it + eps_it

    Same selection as SelectionTimingDGP, but group-specific trends
    theta_g violate parallel trends.
    """

    def __init__(self, N: int = 100, T: int = 10,
                 g_early: int = 4, g_late: int = 7,
                 share_never: float = 0.40,
                 share_early: float = 0.30,
                 share_late: float = 0.30,
                 mu_b: float = 1.0, sigma_b: float = 0.5,
                 trend_early: float = 0.3,
                 trend_late: float = 0.1,
                 trend_never: float = 0.0,
                 sigma_alpha: float = 1.0, sigma_eps: float = 1.0):
        self.N = N; self.T = T
        self.g_early = g_early; self.g_late = g_late
        self.share_never = share_never
        self.share_early = share_early
        self.share_late = share_late
        self.mu_b = mu_b; self.sigma_b = sigma_b
        self.trend_early = trend_early
        self.trend_late = trend_late
        self.trend_never = trend_never
        self.sigma_alpha = sigma_alpha
        self.sigma_eps = sigma_eps
        self._true_att = mu_b

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        N, T = self.N, self.T

        b = rng.normal(self.mu_b, self.sigma_b, size=N)

        q_never = np.quantile(b, self.share_never)
        q_late = np.quantile(b, self.share_never + self.share_late)

        G_unit = np.zeros(N, dtype=int)
        G_unit[b > q_late] = self.g_early
        G_unit[(b > q_never) & (b <= q_late)] = self.g_late

        N_early = int((G_unit == self.g_early).sum())
        N_late = int((G_unit == self.g_late).sum())
        self._true_att = float(
            (N_early * b[G_unit == self.g_early].mean() +
             N_late * b[G_unit == self.g_late].mean()) / (N_early + N_late)
        )

        alpha = rng.normal(0.0, self.sigma_alpha, size=N)

        trend_map = {self.g_early: self.trend_early,
                    self.g_late: self.trend_late,
                    0: self.trend_never}
        theta_unit = np.array([trend_map[g] for g in G_unit])

        id_panel = np.repeat(np.arange(1, N + 1), T)
        t_panel = np.tile(np.arange(1, T + 1), N)
        G_panel = np.repeat(G_unit, T)
        D_panel = ((G_panel > 0) & (t_panel >= G_panel)).astype(int)
        alpha_panel = np.repeat(alpha, T)
        b_panel = np.repeat(b, T)
        theta_panel = np.repeat(theta_unit, T)
        lambda_panel = t_panel.astype(np.float64)
        eps_panel = rng.normal(0.0, self.sigma_eps, size=N * T)

        group_trend = theta_panel * t_panel
        Y_panel = alpha_panel + lambda_panel + group_trend + b_panel * D_panel + eps_panel

        df = pd.DataFrame({
            "id": id_panel,
            "t": t_panel,
            "G": G_panel,
            "D": D_panel,
            "alpha": alpha_panel,
            "lambda_t": lambda_panel,
            "group_trend": group_trend,
            "b_i": b_panel,
            "eps": eps_panel,
            "Y": Y_panel,
        })
        return df

    @property
    def true_att(self) -> float:
        return self._true_att
