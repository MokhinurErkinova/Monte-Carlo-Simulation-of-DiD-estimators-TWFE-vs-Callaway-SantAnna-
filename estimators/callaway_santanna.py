"""
estimators/callaway_santanna.py — Callaway-Sant'Anna (2021) estimator.

Contains all CS estimation logic including:
- ATT(g,t) computation using not-yet-treated controls
- Simple aggregation (cohort-size weighted average)

References
----------
Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

import numpy as np
import pandas as pd
from typing import Tuple

class CallawaySantannaEstimator:
    """
    Callaway-Sant'Anna (2021) difference-in-differences estimator.

    Computes group-time average treatment effects ATT(g,t) using only
    not-yet-treated units as controls, then aggregates to an overall ATT.

    This avoids the "forbidden comparisons" problem in TWFE where already-treated
    units serve as implicit controls for later-treated units.

    Attributes
    ----------
    estimate : float
        Aggregated ATT (after calling fit)
    attgt_df : pd.DataFrame
        Group-time ATT estimates (after calling fit)
    n_skipped : int
        Number of (g,t) cells skipped due to no valid controls
    """

    def __init__(self):
        self._estimate = np.nan
        self._attgt_df = None
        self._n_skipped = 0

    def fit(self, df: pd.DataFrame, y: str = "Y", unit: str = "id",
            time: str = "t", gcol: str = "G") -> None:
        """
        Fit CS estimator to panel data.

        Parameters
        ----------
        df : pd.DataFrame
            Balanced panel with columns for outcome, unit ID, time, and cohort.
            Must be sorted by (unit, time).
        y : str
            Name of outcome column
        unit : str
            Name of unit identifier column
        time : str
            Name of time period column
        gcol : str
            Name of cohort column (G=0 for never-treated, G>0 for treatment time)
        """
        self._attgt_df, self._n_skipped = self._compute_attgt(df, y, unit, time, gcol)
        self._estimate = self._aggregate(self._attgt_df)

    def _compute_attgt(self, df: pd.DataFrame, y: str, unit: str,
                      time: str, gcol: str) -> Tuple[pd.DataFrame, int]:
        """
        Compute ATT(g,t) for every (cohort, post-period) using not-yet-treated controls.

        Uses matrix operations for efficiency: pivots Y into an (N × T) matrix once,
        then each ATT(g,t) is computed via boolean-mask indexing.
        """
        T = int(df[time].max())
        G_arr = df[gcol].values[::T]
        N = len(G_arr)

        Y_mat = df[y].values.reshape(N, T)

        cohorts = sorted(set(G_arr[G_arr > 0]))
        cohort_mask = {g: (G_arr == g) for g in cohorts}
        never_mask = (G_arr == 0)

        rows = []
        n_skipped = 0

        for g in cohorts:
            base_col = g - 2
            if base_col < 0:
                raise ValueError(f"Cohort g={g} has baseline g-1={g-1}<1. Choose g>=2.")

            treated = cohort_mask[g]
            n_treat = int(treated.sum())

            if n_treat == 0:
                continue

            Y_treat_base = Y_mat[treated, base_col].mean()

            for t_val in range(g, T + 1):
                t_col = t_val - 1

                control = never_mask | (G_arr > t_val)
                n_ctrl = int(control.sum())

                if n_ctrl == 0:
                    n_skipped += 1
                    continue

                Y_treat_t = Y_mat[treated, t_col].mean()
                Y_ctrl_t = Y_mat[control, t_col].mean()
                Y_ctrl_base = Y_mat[control, base_col].mean()

                att = (Y_treat_t - Y_treat_base) - (Y_ctrl_t - Y_ctrl_base)

                rows.append({
                    "g": g,
                    "t": t_val,
                    "att_gt": float(att),
                    "n_treated": n_treat,
                    "n_control": n_ctrl,
                })

        return pd.DataFrame(rows), n_skipped

    def _aggregate(self, attgt_df: pd.DataFrame) -> float:
        """
        Aggregate ATT(g,t) to overall ATT using cohort-size weights.

        Computes: weighted average of within-cohort mean ATT(g,t),
        where weights are proportional to cohort sizes.
        """
        if attgt_df.empty:
            return np.nan

        by_g = (attgt_df
                .groupby("g")
                .agg({"att_gt": "mean", "n_treated": "first"})
                .reset_index())
        by_g.columns = ["g", "att_g", "n_g"]

        w = by_g["n_g"] / by_g["n_g"].sum()
        return float((w * by_g["att_g"]).sum())

    @property
    def estimate(self) -> float:
        """Aggregated ATT estimate."""
        return self._estimate

    @property
    def attgt_df(self) -> pd.DataFrame:
        """DataFrame of group-time ATT estimates."""
        return self._attgt_df

    @property
    def n_skipped(self) -> int:
        """Number of (g,t) cells skipped due to no valid controls."""
        return self._n_skipped

    @property
    def name(self) -> str:
        """Estimator name for reporting."""
        return "CS"
