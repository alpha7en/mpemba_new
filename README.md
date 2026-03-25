# Quantum Dynamics Research Code

This repository contains computational code for a study of dissipative single-photon dynamics on rewired grids.
The core objective is to analyze how network topology affects relaxation modes, spectral gaps, and localization metrics.

## Project dataflow

The workflow is split into two stages:

1. **Heavy spectral precomputation**
   - `scripts/run_multicore_npz.py`
   - runs expensive sparse simulations and stores results in a large `.npz` archive
2. **Fast metric/figure analysis from `.npz`**
   - scripts such as `scripts/fig6_lambda_visualization.py` and `scripts/fig7_ipr_visualization.py`
   - reuse the precomputed archive to build statistical plots quickly

This separation is intentional: long simulation is executed rarely, while analysis scripts are run repeatedly.

## Code structure

- `src/qdyn_research/`
  - scientific kernels: topology generation, Liouvillian builders, spectral solvers, metrics, ODE utilities
- `scripts/`
  - runnable figure/analysis scripts
- `tests/`
  - smoke and regression checks for core behavior
- `docs/`
  - method inventory and figure coverage notes

## Main scripts

- `scripts/run_multicore_npz.py` — multicore sparse spectral batch, writes `.npz`
- `scripts/fig1_grouped_modes.py` — grouped mode population maps
- `scripts/fig2_projection_bars_and_modes.py` — modal projection bars + mode maps
- `scripts/fig3_relaxation_dynamics.py` — relaxation trajectories and crossings
- `scripts/fig4_average_path_length.py` — topological metric `<L>(p)`
- `scripts/fig5_topology_and_spectrum.py` — topology plus spectral panel
- `scripts/fig6_lambda_visualization.py` — mode-wise `Re(λ_k)` statistics from `.npz`
- `scripts/fig7_ipr_visualization.py` — IPR statistics from `.npz`
- `scripts/fig9_task_b_heatmap.py` — excitability-map composite panel
- `scripts/fig10_random_benchmark.py` — random-vs-targeted benchmark in `res_gap/`
- `scripts/fig11_algorithm_visual.py` — stepwise hot/cold selection visualization
- `scripts/first_mode_analysis.py` — first slow-mode diagnostic helper

## Install

```bash
cd /Users/alpha7en/PycharmProjects/alod2/research_refactor
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick check

```bash
cd /Users/alpha7en/PycharmProjects/alod2/research_refactor
python -m pytest -q
```

## Run heavy simulation stage

```bash
cd /Users/alpha7en/PycharmProjects/alod2/research_refactor
python scripts/run_multicore_npz.py
```

## Run analysis on an existing `.npz`

```bash
cd /Users/alpha7en/PycharmProjects/alod2/на\ проверку
python /Users/alpha7en/PycharmProjects/alod2/research_refactor/scripts/fig6_lambda_visualization.py
python /Users/alpha7en/PycharmProjects/alod2/research_refactor/scripts/fig7_ipr_visualization.py
```

## Run figure scripts directly

```bash
cd /Users/alpha7en/PycharmProjects/alod2/research_refactor
python scripts/fig1_grouped_modes.py
python scripts/fig2_projection_bars_and_modes.py
python scripts/fig3_relaxation_dynamics.py
python scripts/fig4_average_path_length.py
python scripts/fig5_topology_and_spectrum.py
python scripts/fig9_task_b_heatmap.py
python scripts/fig10_random_benchmark.py
python scripts/fig11_algorithm_visual.py
python scripts/first_mode_analysis.py
```

## Scientific conventions (short)

- Liouvillian and mode analysis follow the matrix-vectorized Lindblad form.
- `order='F'` vectorization is used where required by Kronecker-structured operators.
- Some entropy-analysis paths use `order='C'` by design in scripts that operate on diagonal-state vectors.
