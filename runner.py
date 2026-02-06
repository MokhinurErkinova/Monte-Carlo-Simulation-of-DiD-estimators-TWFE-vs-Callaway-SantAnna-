
#runner.py — Monte Carlo simulation runner.


import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Summary statistics from a Monte Carlo run."""
    estimator_name: str
    mean: float
    bias: float
    sd: float
    rmse: float
    n_reps: int
    true_att: float

    def __str__(self):
        return (f"Mean={self.mean:.4f}, Bias={self.bias:.4f}, "
                f"SD={self.sd:.4f}, RMSE={self.rmse:.4f}, N={self.n_reps}")

class SimulationRunner:
    """
    Monte Carlo simulation runner for DiD estimators.

    Pairs a DGP with an estimator and runs R replications, computing
    summary statistics (mean, bias, SD, RMSE).

    Parameters
    ----------
    dgp : DGPProtocol
        Data generating process with .sample(seed) and .true_att
    estimator : EstimatorProtocol
        Estimator with .fit(df) and .estimate

    Example
    -------
    >>> from dgps.staggered_dgp import StaggeredDGP
    >>> from estimators.twfe import TWFEEstimator
    >>> dgp = StaggeredDGP(N0=40, N_early=30, N_late=30, T=10)
    >>> runner = SimulationRunner(dgp, TWFEEstimator())
    >>> result = runner.simulate(n_sim=1000, first_seed=42)
    >>> print(result)
    """

    def __init__(self, dgp, estimator):
        self.dgp = dgp
        self.estimator = estimator

    def simulate(self, n_sim: int = 1000, first_seed: int = 1000,
                 verbose: bool = False) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        n_sim : int
            Number of replications
        first_seed : int
            Starting seed (incremented by 1 for each replication)
        verbose : bool
            Print progress updates

        Returns
        -------
        SimulationResult
            Summary statistics across replications
        """
        estimates = []
        truths = []

        for r in range(n_sim):
            if verbose and n_sim > 1 and (r + 1) % 1000 == 0:
                print(f"  {r + 1}/{n_sim} replications done...")

            df = self.dgp.sample(seed=first_seed + r)
            truths.append(self.dgp.true_att)
            self.estimator.fit(df)
            estimates.append(self.estimator.estimate)

        return self._summarize(estimates, truths)

    @staticmethod
    def _summarize(estimates, truths) -> SimulationResult:
        """Compute summary statistics from estimates and truths."""
        est = np.array(estimates, dtype=float)
        tru = np.array(truths, dtype=float)

        mask = ~np.isnan(est)
        est = est[mask]
        tru = tru[mask]

        if len(est) == 0:
            raise ValueError("All estimates are NaN.")

        mean_est = float(est.mean())
        mean_tru = float(tru.mean())
        diff = est - tru
        bias = float(diff.mean())
        sd = float(est.std(ddof=1)) if len(est) > 1 else 0.0
        rmse = float(np.sqrt((diff ** 2).mean()))

        return SimulationResult(
            estimator_name="",
            mean=mean_est,
            bias=bias,
            sd=sd,
            rmse=rmse,
            n_reps=len(est),
            true_att=mean_tru,
        )
