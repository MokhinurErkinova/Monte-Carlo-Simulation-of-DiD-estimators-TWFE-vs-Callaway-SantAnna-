"""
main.py — Master orchestrator for the DiD simulation study.

Compares TWFE vs Callaway-Sant'Anna estimators across multiple scenarios
designed to demonstrate when and how TWFE breaks down.

All scenario parameters are defined in the SCENARIO_REGISTRY.
"""

import time
import os
import pandas as pd

from dgps.deterministic_dgp import DeterministicDGP
from dgps.static_dgp import StaticNormalDGP
from dgps.heterogeneous_dgp import HeterogeneousDGP
from dgps.staggered_dgp import StaggeredDGP
from dgps.selection_dgp import SelectionTimingDGP, SelectionPretrendsDGP

from estimators.twfe import TWFEEstimator
from estimators.callaway_santanna import CallawaySantannaEstimator

from runner import SimulationRunner

# ==============================================================
# SCENARIO REGISTRY
# ==============================================================

def _build_registry():
    """Return ordered list of (label, dgp, n_sim, first_seed) tuples."""
    gs = {"treated_A": 30, "treated_B": 30, "treated_C": 30, "control": 40}
    ge = {"treated_A": 0.5, "treated_B": 1.0, "treated_C": 1.5, "control": 0.0}
    gs_l = {k: v * 10 for k, v in gs.items()}

    return [
        # ----------------------------------------------------------
        # Stage 0: Deterministic unit test
        # ----------------------------------------------------------
        ("stage0",
         DeterministicDGP(N_treated=50, N_control=50, T=10, g=6, tau=1.0),
         1, 1000),

        # ----------------------------------------------------------
        # Stage 1: Homogeneous effects
        # ----------------------------------------------------------
        ("stage1_std",
         StaticNormalDGP(N_treated=50, N_control=50, T=10, g=6, tau=1.0),
         5000, 1000),

        ("stage1_large",
         StaticNormalDGP(N_treated=500, N_control=500, T=10, g=6, tau=1.0),
         5000, 1000),

        # ----------------------------------------------------------
        # Stage 2: Heterogeneous effects, same timing
        # ----------------------------------------------------------
        ("stage2_std",
         HeterogeneousDGP(group_sizes=gs, group_effects=ge, T=10, g=6),
         5000, 2000),

        ("stage2_large",
         HeterogeneousDGP(group_sizes=gs_l, group_effects=ge, T=10, g=6),
         5000, 2000),

        # ----------------------------------------------------------
        # Stage 3A: Staggered adoption (TWFE breakdown)
        # ----------------------------------------------------------
        ("stage3a_std",
         StaggeredDGP(N0=40, N4=30, N7=30, T=10, tau4=1.5, tau7=0.5),
         2000, 30000),

        ("stage3a_extreme",
         StaggeredDGP(N0=40, N4=20, N7=50, T=15, tau4=5.0, tau7=0.5),
         2000, 30000),

        # ----------------------------------------------------------
        # Stage 3A Nuclear: DYNAMIC effects for SIGN REVERSAL
        # ----------------------------------------------------------
        ("stage3a_nuclear_dynamic",
         StaggeredDGP(N0=20, N_early=100, N_late=30, T=20,
                     g_early=3, g_late=15,
                     tau_early=1.0, tau_late=0.1, tau_early_growth=0.5),
         2000, 30000),

        ("stage3a_nuclear_extreme_dynamic",
         StaggeredDGP(N0=10, N_early=150, N_late=20, T=25,
                     g_early=3, g_late=18,
                     tau_early=2.0, tau_late=0.05, tau_early_growth=1.0),
         2000, 30000),

        # ----------------------------------------------------------
        # Stage 3A Large Samples
        # ----------------------------------------------------------
        ("stage3a_std_large",
         StaggeredDGP(N0=400, N4=300, N7=300, T=10, tau4=1.5, tau7=0.5),
         2000, 30000),

        ("stage3a_extreme_large",
         StaggeredDGP(N0=400, N4=200, N7=500, T=15, tau4=5.0, tau7=0.5),
         2000, 30000),

        ("stage3a_nuclear_dynamic_large",
         StaggeredDGP(N0=200, N_early=1000, N_late=300, T=20,
                     g_early=3, g_late=15,
                     tau_early=1.0, tau_late=0.1, tau_early_growth=0.5),
         2000, 30000),

        # ----------------------------------------------------------
        # Stage 3B: Selection into timing
        # ----------------------------------------------------------
        ("stage3b_random",
         SelectionTimingDGP(N=100, T=10, g_early=4, g_late=7,
                           share_never=0.40, share_early=0.30, share_late=0.30,
                           mu_b=1.0, sigma_b=0.5, b_fixed=None),
         2000, 40000),

        ("stage3b_fixed",
         SelectionTimingDGP(N=100, T=10, g_early=4, g_late=7,
                           share_never=0.40, share_early=0.30, share_late=0.30,
                           mu_b=1.0, sigma_b=0.5, b_fixed="use_fixed"),
         2000, 40000),

        ("stage3b_random_large",
         SelectionTimingDGP(N=1000, T=10, g_early=4, g_late=7,
                           share_never=0.40, share_early=0.30, share_late=0.30,
                           mu_b=1.0, sigma_b=0.5, b_fixed=None),
         2000, 40000),

        ("stage3b_fixed_large",
         SelectionTimingDGP(N=1000, T=10, g_early=4, g_late=7,
                           share_never=0.40, share_early=0.30, share_late=0.30,
                           mu_b=1.0, sigma_b=0.5, b_fixed="use_fixed"),
         2000, 40000),

        ("stage3b_pretrends",
         SelectionPretrendsDGP(N=100, T=10, g_early=4, g_late=7,
                              share_never=0.40, share_early=0.30, share_late=0.30,
                              mu_b=1.0, sigma_b=0.5,
                              trend_early=0.3, trend_late=0.1, trend_never=0.0),
         2000, 40000),

        ("stage3b_pretrends_large",
         SelectionPretrendsDGP(N=1000, T=10, g_early=4, g_late=7,
                              share_never=0.40, share_early=0.30, share_late=0.30,
                              mu_b=1.0, sigma_b=0.5,
                              trend_early=0.3, trend_late=0.1, trend_never=0.0),
         2000, 40000),
    ]


def _get_sample_sizes(dgp):
    """Extract sample size information from a DGP."""
    n_total = n_treated = n_control = n_early = n_late = T_periods = None

    if hasattr(dgp, 'N_treated') and hasattr(dgp, 'N_control'):
        n_treated = dgp.N_treated
        n_control = dgp.N_control
        n_total = n_treated + n_control
        T_periods = dgp.T

    elif hasattr(dgp, 'group_sizes'):
        n_control = dgp.group_sizes.get('control', 0)
        n_treated = sum(v for k, v in dgp.group_sizes.items() if k != 'control')
        n_total = sum(dgp.group_sizes.values())
        T_periods = dgp.T

    elif hasattr(dgp, 'N0') and hasattr(dgp, 'N_early'):
        n_control = dgp.N0
        n_early = dgp.N_early
        n_late = dgp.N_late
        n_treated = n_early + n_late
        n_total = n_control + n_treated
        T_periods = dgp.T

    elif hasattr(dgp, 'N0') and hasattr(dgp, 'N4'):
        n_control = dgp.N0
        n_early = dgp.N4
        n_late = dgp.N7
        n_treated = n_early + n_late
        n_total = n_control + n_treated
        T_periods = dgp.T

    elif hasattr(dgp, 'N'):
        n_total = dgp.N
        T_periods = dgp.T

    return n_total, n_treated, n_control, n_early, n_late, T_periods


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("\n" + "=" * 70)
    print("  DIFFERENCE-IN-DIFFERENCES SIMULATION STUDY")
    print("  Comparing TWFE vs Callaway-Sant'Anna Estimators")
    print("=" * 70)

    registry = _build_registry()
    all_results = []
    t0 = time.time()

    for label, dgp, n_sim, first_seed in registry:
        print(f"\n── {label} (R={n_sim}) ──")

        if hasattr(dgp, 'is_dynamic') and dgp.is_dynamic:
            print(f"  [DYNAMIC EFFECTS: early cohort effect grows over time]")

        n_total, n_treated, n_control, n_early, n_late, T_periods = _get_sample_sizes(dgp)

        twfe_diag = None
        if label.startswith("stage3"):
            sample_df = dgp.sample(seed=first_seed)
            twfe_est_temp = TWFEEstimator()
            twfe_est_temp.fit(sample_df)
            twfe_diag = twfe_est_temp.diagnostics

        for Est in (TWFEEstimator, CallawaySantannaEstimator):
            est = Est()
            runner = SimulationRunner(dgp, est)
            res = runner.simulate(n_sim=n_sim, first_seed=first_seed, verbose=True)
            res.estimator_name = est.name

            sign_reversal = (res.true_att > 0 and res.mean < 0) or \
                           (res.true_att < 0 and res.mean > 0)
            reversal_flag = " 🚨 SIGN REVERSAL!" if sign_reversal else ""

            print(f"  {est.name:>4}: mean={res.mean:.6f} bias={res.bias:.6f} "
                  f"sd={res.sd:.6f} rmse={res.rmse:.6f} (truth={res.true_att:.6f}){reversal_flag}")

            result = {
                "scenario": label,
                "estimator": est.name,
                "mean": res.mean,
                "bias": res.bias,
                "sd": res.sd,
                "rmse": res.rmse,
                "true_att": res.true_att,
                "n_reps": res.n_reps,
                "sign_reversal": sign_reversal,
                "N_total": n_total,
                "N_treated": n_treated,
                "N_control": n_control,
                "N_early": n_early,
                "N_late": n_late,
                "T": T_periods,
                "neg_weight_share": None,
                "n_neg_weights": None,
                "min_weight": None,
                "max_weight": None,
            }

            if est.name == "TWFE" and twfe_diag is not None:
                result["neg_weight_share"] = twfe_diag.prop_treated_negative
                result["n_neg_weights"] = twfe_diag.n_treated_negative
                result["min_weight"] = twfe_diag.min_weight_treated
                result["max_weight"] = twfe_diag.max_weight_treated

            all_results.append(result)

        if twfe_diag is not None:
            print(f"  TWFE neg-weight share: {twfe_diag.prop_treated_negative:.1%}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/simulation_results.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE — {len(all_results)} scenario×estimator pairs in {elapsed:.0f}s")
    print(f"  Results → results/simulation_results.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
