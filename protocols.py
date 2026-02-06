"""
protocols.py - Interface definitions for DGPs and estimators.

Uses typing.Protocol for structural subtyping: classes do NOT need to
explicitly inherit from these; they just need the right methods/properties.
"""
from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class DGPProtocol(Protocol):
    """Protocol for data generating processes."""

    def sample(self, seed: int | None = None) -> pd.DataFrame:
        """Generate one sample dataset.
        Must return a DataFrame with at least: id, t, G, D, Y
        """
        ...

    @property
    def true_att(self) -> float:
        """True ATT.  For DGPs where truth varies per replication
        (e.g. selection DGPs) this should be updated by sample()."""
        ...


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Protocol for difference-in-differences estimators."""

    def fit(self, df: pd.DataFrame) -> None:
        """Fit estimator to panel data with columns: id, t, G, D, Y"""
        ...

    @property
    def estimate(self) -> float:
        """Point estimate of treatment effect."""
        ...

    @property
    def name(self) -> str:
        """Estimator name for reporting."""
        ...
