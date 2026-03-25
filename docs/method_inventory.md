# Method Inventory (Phase 1)

This file captures the first-pass parsing of duplicated scientific kernels from `на проверку/`.

## Canonical groups extracted into `qdyn_research`

- `check_connectivity`:
  - `rewiring_spectrum_data_npz_gemeration_for_figs/multicore_main_simulation.py`
  - `fig_9_10_11/reconecting/bright_dark/bright_dark_math/{dense_core.py,core_dense.py,sparse_core.py}`
- `generate_grid_tau`:
  - `fig_1`, `fig_2`, `fig_3`, `fig_4`, `fig_5`, `multicore_main_simulation.py`, `bright_dark_math/*`
- Rewiring topology:
  - Single-pass (no connectivity guarantee): `fig_4`, `fig_5`
  - Guaranteed connectivity loop: `multicore_main_simulation.py`, `bright_dark_math/*`
- Liouvillian build:
  - Dense: `fig_1`, `fig_2`, `fig_5`, `bright_dark_math/*`
  - Sparse: `multicore_main_simulation.py`, `bright_dark_math/sparse_core.py`
- Spectral solvers:
  - Dense strict left/right: `fig_1`, `fig_2`, `bright_dark_math/*`
  - Sparse robust shift-invert: `multicore_main_simulation.py`
  - Sparse biorthogonal via `L.H`: `bright_dark_math/sparse_core.py`
- Projection coefficient and excitability:
  - `first_mode_analys.py`, `fig_2/core.py`, `bright_dark_math/*`
- Localization/entropy metrics:
  - IPR (`order='F'`): `fig_7`
  - Entropic distance (`first_mode_analys.py`) and `logm` metric (`fig_2/core.py`)
- Time simulation core:
  - `fig_2/fully_simulation_core.py`, `fig_3/fully_simulation_core.py`

## Important differences that must stay explicit

- `entropic_distance` reshape order in `first_mode_analys.py` defaults to C-order.
- Most mode/population operations elsewhere use F-order (`flatten('F')` / `reshape(..., order='F')`).
- Dense eigen decomposition appears in two forms:
  - full vectors only (`np.linalg.eig`) and
  - strict biorthogonal (`scipy.linalg.eig(left=True, right=True)`).
- Rewiring appears in two scientific variants:
  - no connectivity guarantee (topological diagnostics),
  - guaranteed connected graph (spectral statistics workflow).

## Files intentionally not edited

All legacy scripts in `на проверку/` are kept untouched. Refactoring code is isolated in `research_refactor/`.

