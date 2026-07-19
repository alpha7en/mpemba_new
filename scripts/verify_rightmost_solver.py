"""Verification harness for analyze_liouvillian_modes_rightmost (referee fix CF-2).

Before recomputing the Figs. 6-8 sweep with the corrected slowest-mode identification,
this script proves the new solver against dense ground truth:

  6x6 lattice, p in a 13-point grid x 5 seeds (61 graphs): for each graph
    - dense truth: full eig, all 1296 modes, rightmost non-trivial cluster structure;
    - sparse solver: analyze_liouvillian_modes_rightmost (exp-transform Arnoldi, k modes);
    - HARD checks (what the sweep figures actually need):
      (1) the TRUE rightmost mode is present (max-Re match, tol 1e-8);
      (2) the top N_GROUPS DISTINCT eigenvalue groups (complex clustering, tol 1e-6) are
          each represented by at least one solver mode (tol 1e-8). Multiplicity within a
          degenerate group and lambda* conjugate partners are NOT required: single-vector
          Krylov spaces carry one direction per distinct eigenvalue, and the partner's
          diagonal weights are complex conjugates (identical IPR/overlap).
      (3) per-mode residuals <= 1e-8 (enforced inside the solver, re-reported);
      (4) the dense spectrum is conjugate-symmetric (sanity for the Im band).
    - DIAGNOSTIC (reported, not fatal): Hausdorff distance of the top-8 individual-mode
      prefix, which penalizes missing degenerate copies.

Run:  python verify_rightmost_solver.py
Exit code 0 = all graphs pass; details printed per failure.
"""
import sys
import time

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense, build_liouvillian_sparse
from qdyn_research.spectral import analyze_liouvillian_modes_rightmost
from qdyn_research.topology import (
    generate_grid_tau,
    generate_rewired_grid_tau_guaranteed_connectivity,
)

H = W = 6
J, GAMMA = 1.0, 0.1
K_MODES = 20                 # modes requested from the solver (incl. stationary)
N_GROUPS = 4                 # distinct rightmost eigenvalue groups that MUST be found
N_COMPARE = 8                # individual-mode prefix for the (non-fatal) Hausdorff diagnostic
GROUP_TOL = 1e-6             # complex clustering tolerance for "distinct group"
P_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def dense_truth(tau):
    L = build_liouvillian_dense(tau, J, GAMMA)
    ev = np.linalg.eigvals(L)
    return ev[np.argsort(ev.real)[::-1]]


def distinct_groups(values, tol):
    """Representative value of each distinct complex cluster, in encounter (Re-desc) order."""
    reps = []
    for v in values:
        if not any(abs(v - r) < tol for r in reps):
            reps.append(v)
    return reps


def check_graph(tau, label):
    ev_dense = dense_truth(tau)
    nt_dense = ev_dense[1:]                       # non-trivial (stationary dropped)

    # conjugate symmetry of the dense spectrum (pairing tolerance)
    conj_dist = np.abs(nt_dense[:, None] - np.conj(nt_dense)[None, :]).min(axis=1).max()

    Ls = build_liouvillian_sparse(tau, J, GAMMA)
    lam, _vecs, resid = analyze_liouvillian_modes_rightmost(Ls, num_modes=K_MODES)
    if lam is None:
        return [f"{label}: solver returned None (Arnoldi failure or residual violation)"], ""

    # drop the stationary mode from the solver output
    lam_nt = lam[np.abs(lam) > 1e-10]
    if len(lam_nt) < N_GROUPS:
        return [f"{label}: only {len(lam_nt)} non-trivial modes returned"], ""

    fails = []
    # (1) the single TRUE rightmost mode must be found
    d_right = np.abs(lam_nt - nt_dense[0]).min()
    if d_right > 1e-8:
        fails.append(f"{label}: rightmost mode missed (nearest {d_right:.2e}; "
                     f"true {nt_dense[0]:.6f})")
    # (2) every one of the top N_GROUPS distinct dense groups must be represented
    groups = distinct_groups(nt_dense[:60], GROUP_TOL)[:N_GROUPS]
    for gi, g in enumerate(groups):
        d = np.abs(lam_nt - g).min()
        if d > 1e-8:
            fails.append(f"{label}: distinct group #{gi+1} (lambda={g:.6f}) missed by {d:.2e}")
    # (3) residuals (already enforced inside the solver; re-reported)
    if resid.max() > 1e-8:
        fails.append(f"{label}: residual {resid.max():.2e}")
    # (4) conjugate symmetry sanity
    if conj_dist > 1e-9:
        fails.append(f"{label}: dense spectrum not conjugate-symmetric ({conj_dist:.2e})")

    # non-fatal diagnostic: individual-mode top-8 prefix completeness
    lam_top = lam_nt[:N_COMPARE]
    tru_top = nt_dense[:N_COMPARE]
    dist = np.abs(lam_top[:, None] - tru_top[None, :])
    d_haus = max(dist.min(axis=1).max(), dist.min(axis=0).max())
    diag = f"prefix-Hausdorff={d_haus:.1e}"
    return fails, diag


def main():
    t0 = time.time()
    all_fails = []
    n_checked = 0
    for p in P_GRID:
        for seed in SEEDS:
            if p == 0.0:
                if seed > 0:
                    continue
                tau = generate_grid_tau(H, W)
            else:
                tau = generate_rewired_grid_tau_guaranteed_connectivity(H, W, p, seed=seed)
            label = f"p={p:<6g} seed={seed}"
            fails, diag = check_graph(tau, label)
            n_checked += 1
            status = "PASS" if not fails else "FAIL"
            print(f"  {status}  {label}  {diag}  ({time.time()-t0:.0f}s)", flush=True)
            all_fails += fails

    print(f"\nchecked {n_checked} graphs in {time.time()-t0:.0f}s")
    if all_fails:
        print(f"FAILURES ({len(all_fails)}):")
        for f in all_fails:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS: rightmost-mode identification verified against dense truth")


if __name__ == "__main__":
    main()
