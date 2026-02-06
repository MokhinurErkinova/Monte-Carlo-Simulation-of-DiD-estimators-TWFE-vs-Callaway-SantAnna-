# Monte-Carlo-Simulation-of-DiD-estimators-TWFE-vs-Callaway-SantAnna
A Monte Carlo simulation study comparing Two-Way Fixed Effects (TWFE) and Callaway-SantAnna (2021) estimators across multiple difference-in-differences scenarios.
# Difference-in-Differences Monte Carlo Simulation Study

A simulation framework comparing **Two-Way Fixed Effects (TWFE)** and **Callaway-Sant'Anna (2021)** estimators under various data generating processes, with particular focus on demonstrating TWFE breakdown in staggered adoption designs.

## Project Structure

```
did_simulation/
├── main.py                     # Master orchestrator - runs all scenarios
├── runner.py                   # Monte Carlo simulation engine
├── protocols.py                # Interface definitions (DGP/Estimator protocols)
│
├── dgps/                       # Data Generating Processes
│   ├── __init__.py
│   ├── deterministic_dgp.py    # Stage 0: Unit test (no noise)
│   ├── static_dgp.py           # Stage 1: Homogeneous effects
│   ├── heterogeneous_dgp.py    # Stage 2: Heterogeneous effects, same timing
│   ├── staggered_dgp.py        # Stage 3A: Staggered adoption (TWFE breakdown)
│   └── selection_dgp.py        # Stage 3B: Selection into timing
│
├── estimators/                 # DiD Estimators
│   ├── __init__.py
│   ├── twfe.py                 # TWFE estimator + weight diagnostics
│   └── callaway_santanna.py    # CS estimator + ATT(g,t) computation
│
├── results/                    # Output directory
└── README.md
```

### Code Organization


- **`estimators/twfe.py`**: Contains `TWFEEstimator` class with:
  - Point estimation via double-demeaning
  - `TWFEDiagnostics` dataclass for negative weight analysis
  - `compute_twfe_weights()` convenience function

- **`estimators/callaway_santanna.py`**: Contains `CallawaySantannaEstimator` class with:
  - ATT(g,t) computation using not-yet-treated controls
  - Cohort-size weighted aggregation
  - Access to detailed group-time estimates via `.attgt_df`


## Quick Start

```bash
# Run the full simulation study
python main.py

# Results saved to results/simulation_results.csv
```

## Simulation Stages

### Stage 0: Deterministic Unit Test
- **DGP**: Y_it = λ_t + τ·D_it (no unit FE, no noise)
- **Purpose**: To verify both estimators recover τ exactly
- **Expected**: TWFE = CS = τ

### Stage 1: Homogeneous Effects
- **DGP**: Y_it = α_i + λ_t + τ·D_it + ε_it
- **Purpose**: Baseline with single treatment timing, constant effect
- **Expected**: Both estimators unbiased

### Stage 2: Heterogeneous Effects (Same Timing)
- **DGP**: Y_it = α_i + λ_t + τ_group·D_it + ε_it
- **Purpose**: Multiple treatment groups, all treated at same time
- **Expected**: Both estimators unbiased (no forbidden comparisons)

### Stage 3A: Staggered Adoption ⚠️ **TWFE Breaks Down**
- **DGP**: Y_it = α_i + λ_t + τ_g(e)·D_it + ε_it
- **Purpose**: Demonstrate negative weighting problem
- **Key scenarios**:
  - `stage3a_std`: Standard staggered design
  - `stage3a_extreme`: Higher effect heterogeneity
  - `stage3a_nuclear_dynamic`: **SIGN REVERSAL** with dynamic effects
  - `stage3a_nuclear_extreme_dynamic`: Maximum sign reversal

### Stage 3B: Selection into Timing
- **DGP**: Selection on treatment gains (b_i)
- **Purpose**: Test robustness to selection mechanisms
- **Variants**: With/without parallel trends violations

## TWFE Sign Reversal

### The Problem
TWFE can estimate a **negative** treatment effect even when **all true effects are positive**.

### When Does This Happen?

| Effect Type | Max Bias | Sign Reversal |
|-------------|----------|---------------|
| Constant    | +1000%+  | ❌ Impossible |
| Dynamic     | -170%    | ✅ Achieved   |

### Why?
With **dynamic effects** (effect grows over time):
1. Early cohort accumulates large treatment effects
2. TWFE uses early-treated as implicit "controls" for late cohort
3. Over-subtraction of elevated early-treated Y → negative bias → sign flip

### Example Configuration
```python
from dgps.staggered_dgp import StaggeredDGP
from estimators.twfe import TWFEEstimator

# This achieves sign reversal
dgp = StaggeredDGP(
    N0=20,              # Never-treated
    N_early=100,        # Early cohort (treated at t=3)
    N_late=30,          # Late cohort (treated at t=15)
    T=20,               # Time periods
    g_early=3,          # Early treatment time
    g_late=15,          # Late treatment time
    tau_early=1.0,      # Base effect for early cohort
    tau_late=0.1,       # Effect for late cohort
    tau_early_growth=0.5  # Effect grows 0.5 per period!
)

df = dgp.sample(seed=42)
est = TWFEEstimator()
est.fit(df)

print(f"True ATT: {dgp.true_att:.4f}")      # ≈ 4.78 (POSITIVE)
print(f"TWFE estimate: {est.estimate:.4f}") # ≈ -0.73 (NEGATIVE!) 🚨
print(est.diagnostics)                       # Shows 30% negative weights
```

## Usage Examples

### Running Specific Scenarios
```python
from dgps.staggered_dgp import StaggeredDGP
from estimators.twfe import TWFEEstimator
from estimators.callaway_santanna import CallawaySantannaEstimator
from runner import SimulationRunner

# Create DGP
dgp = StaggeredDGP(N0=40, N_early=30, N_late=30, T=10,
                   tau_early=1.5, tau_late=0.5)

# Run simulation for each estimator
for Est in [TWFEEstimator, CallawaySantannaEstimator]:
    runner = SimulationRunner(dgp, Est())
    result = runner.simulate(n_sim=1000, first_seed=42)
    print(f"{Est().name}: bias={result.bias:.4f}, RMSE={result.rmse:.4f}")
```

### Accessing TWFE Weight Diagnostics
```python
from estimators.twfe import TWFEEstimator, compute_twfe_weights

# Method 1: Via estimator object
est = TWFEEstimator()
est.fit(df)
print(est.diagnostics)  # TWFEDiagnostics object

# Method 2: Convenience function
diag = compute_twfe_weights(df)
print(f"Negative weight share: {diag.prop_treated_negative:.1%}")
```

### Accessing CS Group-Time Estimates
```python
from estimators.callaway_santanna import CallawaySantannaEstimator

cs = CallawaySantannaEstimator()
cs.fit(df)

print(f"Aggregated ATT: {cs.estimate:.4f}")
print(f"Skipped cells: {cs.n_skipped}")
print(cs.attgt_df)  # DataFrame with ATT(g,t) for each cohort-time
```

## Parameter Reference

### StaggeredDGP Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `N0` | Never-treated units | 40 |
| `N_early` (or `N4`) | Early cohort units | 30 |
| `N_late` (or `N7`) | Late cohort units | 30 |
| `T` | Time periods | 10 |
| `g_early` | Early treatment time | 4 |
| `g_late` | Late treatment time | 7 |
| `tau_early` (or `tau4`) | Early cohort base effect | 1.5 |
| `tau_late` (or `tau7`) | Late cohort effect | 0.5 |
| `tau_early_growth` | **Per-period effect growth** | 0.0 |
| `sigma_alpha` | Unit FE std dev | 1.0 |
| `sigma_eps` | Idiosyncratic error std dev | 1.0 |

## Output Format

Results are saved to `results/simulation_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `scenario` | Scenario identifier |
| `estimator` | TWFE or CS |
| `mean` | Mean estimate across replications |
| `bias` | Mean bias (estimate - truth) |
| `sd` | Standard deviation of estimates |
| `rmse` | Root mean squared error |
| `true_att` | True average treatment effect |
| `n_reps` | Number of valid replications |
| `sign_reversal` | Boolean flag for sign reversal |
| **Sample Sizes** | |
| `N_total` | Total number of units |
| `N_treated` | Number of ever-treated units |
| `N_control` | Number of never-treated units |
| `N_early` | Early cohort size (Stage 3A only) |
| `N_late` | Late cohort size (Stage 3A only) |
| `T` | Number of time periods |
| **TWFE Diagnostics** | |
| `neg_weight_share` | Share of treated obs with negative weights (TWFE only) |
| `n_neg_weights` | Count of negative-weighted treated obs (TWFE only) |
| `min_weight` | Minimum weight among treated obs (TWFE only) |
| `max_weight` | Maximum weight among treated obs (TWFE only) |

## Dependencies

- Python 3.13
- NumPy
- Pandas

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964-2996.

