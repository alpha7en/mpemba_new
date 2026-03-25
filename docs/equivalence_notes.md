# Equivalence Notes (Practical)

This project preserves legacy behavior by keeping explicit strategy variants where scripts differ.

## Strategy split kept intentionally

- `generate_rewired_grid_tau(...)`:
  - single-pass rewiring without enforced connectivity (used by topology plots),
  - `generate_rewired_grid_tau_guaranteed_connectivity(...)` with BFS rejection loop (used by spectral/statistical pipelines).
- Entropy distance:
  - `entropic_distance(..., reshape_order='C')` for compatibility with `fig_9_10_11/first_mode_analys.py`,
  - Fortran-order reshape in Liouvillian-driven modal population extraction (`order='F'`).
- Spectral extraction:
  - Dense biorthogonal (`scipy.linalg.eig(left=True, right=True)`),
  - Dense plain (`np.linalg.eig`) for spectrum-only visual panels,
  - Sparse robust shift-invert for heavy NPZ production.

## Data compatibility

- The generated NPZ keys follow legacy schema:
  - `p_{p}_run_{run_idx}_lambdas`
  - `p_{p}_run_{run_idx}_vectors`
  - `p_{p}_run_{run_idx}_p_value`

## Regression guardrails implemented

- Smoke tests for topology, Liouvillian dimensions, projection/IPR.
- Regression checks for dephasing diagonal indices and guaranteed connectivity.
- Syntax compilation over all library and runnable scripts.

