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

Implemented as a pytest suite in `tests/` (fast set: `python -m pytest -q`; heavy
verification set: `python -m pytest -q -m slow`):

- Smoke tests for topology (seeded reproducibility, guaranteed connectivity, edge counts),
  Liouvillian dimensions and the J-convention (generator == Lindbladian with H = -(1/2)tau,
  exact), dephasing diagonal indices, exact spectral projections (CF-1 control), IPR limits,
  basis invariance of the group diagonal weight (Figs. 7-8 metric).
- Artifact consistency: canonical paper-graph fingerprints, checkpoint/graph agreement,
  bit-reproduction of the stored benchmark crossings in the resolved zone (SEED=7).
- Slow suite: full rebuild_validator 5-check verification; rightmost-mode solver vs dense
  truth on the regular 6x6 lattice (the CF-2 failure mode).

