"""
dgps/staggered_dgp.py — Stage 3A: Staggered adoption, heterogeneous cohort effects.

DGP: Y_it = alpha_i + lambda_t + tau_g(e) * D_it + eps_it

Cohorts: G=0 (never), G=g_early (early, effect tau_early), G=g_late (late, effect tau_late).

When tau_early_growth > 0, the early cohort treatment effect grows over time:
tau_early(e) = tau_early + tau_early_growth * e, where e = periods since treatment.

Panel is emitted in (id, t) order.
"""

import numpy as np
import pandas as pd
from typing import Optional

class StaggeredDGP:
    """
    Staggered adoption DGP with heterogeneous (and optionally dynamic) cohort effects.

    Parameters
    ----------
    N0, N_early, N_late : int
        Number of units in never-treated, early-treated, and late-treated cohorts.
    T : int
        Number of time periods.
    g_early, g_late : int
        Treatment timing for early and late cohorts.
    tau_early, tau_late : float
        Base treatment effects for each cohort.
    tau_early_growth : float, default=0.0
        Per-period growth in early cohort's treatment effect.
        If > 0, tau_early(e) = tau_early + tau_early_growth * e.
    sigma_alpha, sigma_eps : float
        Standard deviations for unit fixed effects and idiosyncratic errors.
    """

    def __init__(
        self,
        N0: int = 40,
        N_early: int = 30,
        N_late: int = 30,
        T: int = 10,
        g_early: int = 4,
        g_late: int = 7,
        tau_early: float = 1.5,
        tau_late: float = 0.5,
        tau_early_growth: float = 0.0,
        sigma_alpha: float = 1.0,
        sigma_eps: float = 1.0,
        N4: Optional[int] = None,
        N7: Optional[int] = None,
        tau4: Optional[float] = None,
        tau7: Optional[float] = None,
    ):
        if N4 is not None:
            N_early = N4
        if N7 is not None:
            N_late = N7
        if tau4 is not None:
            tau_early = tau4
        if tau7 is not None:
            tau_late = tau7

        self.N0 = N0
        self.N_early = N_early
        self.N_late = N_late
        self.T = T
        self.g_early = g_early
        self.g_late = g_late
        self.tau_early = tau_early
        self.tau_late = tau_late
        self.tau_early_growth = tau_early_growth
        self.sigma_alpha = sigma_alpha
        self.sigma_eps = sigma_eps

        self.N4 = N_early
        self.N7 = N_late
        self.tau4 = tau_early
        self.tau7 = tau_late

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        T = self.T
        N = self.N0 + self.N_early + self.N_late

        G_unit = np.concatenate([
            np.zeros(self.N0, dtype=int),
            np.full(self.N_early, self.g_early, dtype=int),
            np.full(self.N_late, self.g_late, dtype=int),
        ])
        alpha = rng.normal(0.0, self.sigma_alpha, size=N)

        id_panel = np.repeat(np.arange(1, N + 1), T)
        t_panel = np.tile(np.arange(1, T + 1), N)
        G_panel = np.repeat(G_unit, T)
        D_panel = ((G_panel > 0) & (t_panel >= G_panel)).astype(int)
        alpha_panel = np.repeat(alpha, T)
        lambda_panel = t_panel.astype(np.float64)
        eps_panel = rng.normal(0.0, self.sigma_eps, size=N * T)

        if self.tau_early_growth == 0:
            tau_base = np.concatenate([
                np.zeros(self.N0),
                np.full(self.N_early, self.tau_early),
                np.full(self.N_late, self.tau_late),
            ])
            tau_panel = np.repeat(tau_base, T)
        else:
            tau_panel = np.zeros(N * T)
            for i in range(N * T):
                g = G_panel[i]
                t = t_panel[i]
                d = D_panel[i]
                if d == 1:
                    if g == self.g_early:
                        e = t - self.g_early
                        tau_panel[i] = self.tau_early + self.tau_early_growth * e
                    elif g == self.g_late:
                        tau_panel[i] = self.tau_late

        Y_panel = alpha_panel + lambda_panel + tau_panel * D_panel + eps_panel

        df = pd.DataFrame({
            "id": id_panel,
            "t": t_panel,
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
        """
        Cohort-size weighted true ATT.
        For dynamic effects, averages over all post-treatment periods.
        """
        if self.tau_early_growth == 0:
            return (self.N_early * self.tau_early + self.N_late * self.tau_late) / \
                   (self.N_early + self.N_late)
        else:
            post_early = self.T - self.g_early + 1
            post_late = self.T - self.g_late + 1

            avg_tau_early = self.tau_early + self.tau_early_growth * (post_early - 1) / 2

            total_early = self.N_early * post_early
            total_late = self.N_late * post_late
            return (total_early * avg_tau_early + total_late * self.tau_late) / \
                   (total_early + total_late)

    @property
    def is_dynamic(self) -> bool:
        """Whether this DGP has dynamic (time-varying) treatment effects."""
        return self.tau_early_growth > 0

    def describe(self) -> str:
        """Return a description of the DGP configuration."""
        lines = [
            f"Staggered DGP Configuration:",
            f"  Cohorts: N0={self.N0}, N_early={self.N_early}, N_late={self.N_late}",
            f"  Timing: g_early={self.g_early}, g_late={self.g_late}, T={self.T}",
        ]

        if self.tau_early_growth > 0:
            tau_final = self.tau_early + self.tau_early_growth * (self.T - self.g_early)
            lines.extend([
                f"  Effects (DYNAMIC):",
                f"    Early: τ(0)={self.tau_early}, growth={self.tau_early_growth}/period, τ({self.T-self.g_early})={tau_final:.1f}",
                f"    Late: τ={self.tau_late} (constant)",
            ])
        else:
            lines.extend([
                f"  Effects (constant): τ_early={self.tau_early}, τ_late={self.tau_late}",
            ])

        lines.append(f"  True ATT: {self.true_att:.4f}")
        return "\n".join(lines)
