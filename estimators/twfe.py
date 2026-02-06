"""
estimators/twfe.py — Two-Way Fixed Effects estimator.

Contains all TWFE estimation logic including:
- Point estimation via double-demeaning
- Weight diagnostics for detecting negative weights
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class TWFEDiagnostics:
    """Diagnostics for TWFE negative weights."""
    prop_treated_negative: float
    n_treated_negative: int
    min_weight_treated: float
    max_weight_treated: float

    def __str__(self):
        return (f"Treated obs w/ negative weights: {self.prop_treated_negative:.1%} "
                f"({self.n_treated_negative} obs)\n"
                f"  Weight range: [{self.min_weight_treated:.6f}, {self.max_weight_treated:.6f}]")

class TWFEEstimator:
    """
    Two-Way Fixed Effects (TWFE) difference-in-differences estimator.

    Estimates the coefficient on the treatment indicator D in:
    Y_it = α_i + λ_t + β·D_it + ε_it

    Uses double-demeaning on (N × T) matrices for computational efficiency.
    Assumes balanced panel sorted by (unit, time).

    Attributes
    ----------
    estimate : float
        Point estimate of treatment effect (after calling fit)
    diagnostics : TWFEDiagnostics
        Weight diagnostics (after calling fit)

    References
    ----------
    - Goodman-Bacon (2021): Difference-in-differences with variation in treatment timing
    - de Chaisemartin & D'Haultfœuille (2020): Two-way fixed effects estimators with
      heterogeneous treatment effects
    """

    def __init__(self):
        self._estimate = np.nan
        self._diagnostics = None

    def fit(self, df: pd.DataFrame, y: str = "Y", d: str = "D",
            unit: str = "id", time: str = "t") -> None:
        """
        Fit TWFE estimator to panel data.

        Parameters
        ----------
        df : pd.DataFrame
            Balanced panel with columns for outcome, treatment, unit ID, and time.
            Must be sorted by (unit, time) with each unit observed in every period.
        y : str
            Name of outcome column
        d : str
            Name of treatment indicator column (0/1)
        unit : str
            Name of unit identifier column
        time : str
            Name of time period column
        """
        N = df[unit].nunique()
        T = int(df[time].max())

        Y_mat = df[y].values.reshape(N, T)
        D_mat = df[d].values.reshape(N, T).astype(np.float64)

        y_i = Y_mat.mean(axis=1, keepdims=True)
        y_t = Y_mat.mean(axis=0, keepdims=True)
        y_bar = Y_mat.mean()

        d_i = D_mat.mean(axis=1, keepdims=True)
        d_t = D_mat.mean(axis=0, keepdims=True)
        d_bar = D_mat.mean()

        y_til = Y_mat - y_i - y_t + y_bar
        d_til = D_mat - d_i - d_t + d_bar

        denom = (d_til ** 2).sum()

        if denom == 0:
            self._estimate = np.nan
            self._diagnostics = TWFEDiagnostics(0.0, 0, 0.0, 0.0)
            return

        self._estimate = float((d_til * y_til).sum() / denom)

        weights = (d_til / denom).ravel()
        D_flat = df[d].values
        treated_w = weights[D_flat == 1]

        if len(treated_w) == 0:
            self._diagnostics = TWFEDiagnostics(0.0, 0, 0.0, 0.0)
        else:
            neg = treated_w < 0
            self._diagnostics = TWFEDiagnostics(
                prop_treated_negative=float(neg.mean()),
                n_treated_negative=int(neg.sum()),
                min_weight_treated=float(treated_w.min()),
                max_weight_treated=float(treated_w.max()),
            )

    @property
    def estimate(self) -> float:
        """Point estimate of treatment effect."""
        return self._estimate

    @property
    def diagnostics(self) -> TWFEDiagnostics:
        """Weight diagnostics from last fit."""
        return self._diagnostics

    @property
    def name(self) -> str:
        """Estimator name for reporting."""
        return "TWFE"

def compute_twfe_weights(df: pd.DataFrame, d: str = "D",
                        unit: str = "id", time: str = "t") -> TWFEDiagnostics:
    """
    Compute TWFE implicit weight diagnostics without full estimation.
    Convenience function for quick diagnostic checks.
    """
    est = TWFEEstimator()
    est.fit(df, d=d, unit=unit, time=time)
    return est.diagnostics
