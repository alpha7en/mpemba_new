"""Rebuild the 6x6 p=0.15 validator checkpoint from the canonical paper-graph artifact.

`precalc/paper_graph_6x6_p015.npz` (small, git-tracked) stores the exact disorder realization
used in the paper (Figs. 11-13, Sec. 8.4 benchmark): the adjacency matrix `tau`, the model
parameters, and physics fingerprints (Liouvillian spectrum, engineered hot/cold nodes, crossing
time t*, initial entropy gap). The heavy checkpoint `precalc/validator_6x6_p015.pkl` (~44 MB,
not in git) is a derived local cache: everything in it is deterministically recomputed from tau.

This script rebuilds the checkpoint and VERIFIES the reconstruction against the stored
fingerprints before writing. Eigenvector phases/scales may differ between LAPACK builds; all
checks therefore target phase-invariant physics (eigenvalues, node selection, t*, gap, D(t)),
which is what the paper actually uses. If any check fails, nothing is written.

Usage:
    python rebuild_validator.py           # writes precalc/validator_6x6_p015.pkl if absent
    python rebuild_validator.py --force   # overwrite an existing checkpoint
    python rebuild_validator.py --out X   # write to a custom path (verification always runs)
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map, entropic_distance
from qdyn_research.mpemba import find_guaranteed_mpemba_dense
from qdyn_research.mpemba_validation import MpembaValidator
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict

PRECALC = Path(__file__).resolve().parent / "precalc"
GRAPH = PRECALC / "paper_graph_6x6_p015.npz"
DEFAULT_OUT = PRECALC / "validator_6x6_p015.pkl"
BENCH = PRECALC / "dirichlet_benchmark_30k.npz"


def rebuild(z):
    """Mirror MpembaValidator.__init__ exactly, but with the stored tau instead of a random one."""
    h, w = int(z["h"]), int(z["w"])
    v = MpembaValidator.__new__(MpembaValidator)
    v.n = h * w
    v.params = {"h": h, "w": w, "p": float(z["p"]), "J": float(z["J"]),
                "gamma": float(z["gamma"]), "seed": None}
    v.tau = z["tau"].copy()
    v.L = build_liouvillian_dense(v.tau, v.params["J"], v.params["gamma"])
    v.evals, v.left_vecs, v.right_vecs = analyze_liouvillian_modes_dense_strict(v.L)
    idx = np.argsort(v.evals.real)[::-1]
    v.evals = v.evals[idx]
    v.left_vecs = v.left_vecs[:, idx]
    v.right_vecs = v.right_vecs[:, idx]
    v.lambda_slow = None
    v.tau_sys = None
    v._compute_spectral_properties()
    # legacy attributes present in the original checkpoint (kept for drop-in compatibility)
    v.N = v.n
    v.max_entropy = float(np.log(v.n))
    return v


def t_cross_vectorized(rho_hot, rho_cold, LV, RV, evals, n):
    """Crossing time on linspace(0,200,400) + linear interpolation — the exact convention of
    run_dirichlet_benchmark.py (verified against the spectral kernel to machine precision)."""
    TP = np.linspace(0.0, 200.0, 400)
    DECAY = np.exp(np.outer(evals, TP))
    DEN = np.sum(LV.conj() * RV, axis=0)
    LOGN = np.log(n)

    def d_curve(rv):
        c = (LV.conj().T @ rv) / DEN
        M = (RV @ (c[:, None] * DECAY)).T.reshape(len(TP), n, n)
        ev = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            return LOGN + np.where(ev > 1e-15, ev * np.log(ev), 0.0).sum(1)

    dh, dc = d_curve(rho_hot), d_curve(rho_cold)
    diff = dc - dh
    idx = np.where(np.diff(np.sign(diff)) > 0)[0]
    i = idx[0]
    return float(TP[i] + (TP[i + 1] - TP[i]) * (-diff[i]) / (diff[i + 1] - diff[i]))


def verify(v, z):
    """Physics-identity checks against the canonical fingerprints. Returns list of failures."""
    fails = []
    n = v.n

    # 1) spectrum (phase-invariant). NOTE: lexicographic sort is unstable for conjugate pairs
    # with nearly equal Re (1e-15 noise flips their order and pairs +i*b with -i*b), so compare
    # the spectra as SETS: symmetric nearest-neighbour (Hausdorff) distance + sorted real parts.
    exp = z["evals_expected"]
    dist = np.abs(v.evals[:, None] - exp[None, :])
    d_haus = max(dist.min(axis=1).max(), dist.min(axis=0).max())
    d_re = np.max(np.abs(np.sort(v.evals.real) - np.sort(exp.real)))
    print(f"  [1] spectrum: Hausdorff = {d_haus:.3e}, sorted-Re max diff = {d_re:.3e}  (tol 1e-8)")
    if d_haus > 1e-8 or d_re > 1e-8:
        fails.append("eigenvalues deviate from the canonical spectrum")

    # 2) engineered pair: discrete node selection must match exactly
    b = calculate_excitability_map(v.left_vecs, v.right_vecs, 1, n)
    vh, _vc, vcs = find_guaranteed_mpemba_dense(v.left_vecs, v.right_vecs, n,
                                                excitability_map=b, mode_idx=1, distance_order="C")
    hot = int(np.argmax(np.diag(vh.reshape(n, n).real)))
    cold = np.sort(np.where(np.diag(vcs.reshape(n, n).real) > 1e-12)[0])
    print(f"  [2] engineered pair: hot={hot} (exp {int(z['hot_idx_expected'])}), "
          f"cold={cold.tolist()} (exp {z['cold_idx_expected'].tolist()})")
    if hot != int(z["hot_idx_expected"]) or not np.array_equal(cold, z["cold_idx_expected"]):
        fails.append("algorithm selects different hot/cold nodes")

    # 3) initial entropy gap of the pair
    gap = float(entropic_distance(vh, n, "C") - entropic_distance(vcs, n, "C"))
    d_gap = abs(gap - float(z["gap_algo_expected"]))
    print(f"  [3] initial gap: {gap:.10f}  (d = {d_gap:.3e}, tol 1e-9)")
    if d_gap > 1e-9:
        fails.append("initial entropy gap deviates")

    # 4) crossing time of the engineered pair
    t_algo = t_cross_vectorized(vh, vcs, v.left_vecs, v.right_vecs, v.evals, n)
    d_t = abs(t_algo - float(z["t_algo_expected"]))
    print(f"  [4] crossing time: t* = {t_algo:.10f}  (d = {d_t:.3e}, tol 1e-4)")
    if d_t > 1e-4:
        fails.append("crossing time deviates")

    # 5) optional cross-check vs stored benchmark data (first admissible Dirichlet pairs).
    # Tolerance model follows the physics: the crossing-time uncertainty grows with t because
    # D(t) decays as exp(-2t/tau) while the eigendecomposition rounding noise stays ~1e-13.
    # Beyond t ~ 150 (D < ~1e-11) crossings are numerically unresolved BY CONSTRUCTION (this is
    # stated in the paper; the figure omits t > 160), so there stored/rebuilt need only agree
    # on the "unresolved" class (t >= T_TAIL or no crossing), not on the noise-driven position.
    if BENCH.exists():
        T_TAIL = 150.0
        d = np.load(BENCH)
        rng = np.random.default_rng(7)                      # SEED of the stored benchmark run
        P = rng.dirichlet(np.ones(n), size=40000)           # BATCH of the stored run
        LOGN = np.log(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = LOGN + np.where(P > 1e-15, P * np.log(P), 0.0).sum(1)
        pa, pb, Da, Db = P[0::2], P[1::2], D[0::2], D[1::2]
        swap = Da < Db
        hot_p = np.where(swap[:, None], pb, pa)
        cold_p = np.where(swap[:, None], pa, pb)
        gap_min = min(LOGN, float(z["gap_algo_expected"])) / 10.0
        keep = (np.where(swap, Db, Da) - np.where(swap, Da, Db)) >= gap_min
        hot_p, cold_p = hot_p[keep], cold_p[keep]
        n_pairs = 12
        stored = d["tstars"][:n_pairs]
        mine = np.array([
            t_cross_or_nan(np.diag(hot_p[k]).astype(complex).flatten("C"),
                           np.diag(cold_p[k]).astype(complex).flatten("C"), v, n)
            for k in range(n_pairs)
        ])
        unresolved = lambda x: np.isnan(x) | (x >= T_TAIL)
        ok_class = np.array_equal(unresolved(mine), unresolved(stored))
        res = ~unresolved(stored)
        d_res = np.max(np.abs(mine[res] - stored[res])) if res.any() else 0.0
        tol_res = 0.2                                       # << histogram bin width (4.0)
        print(f"  [5] benchmark cross-check ({n_pairs} pairs, seed 7): "
              f"resolved (t*<{T_TAIL:.0f}): {int(res.sum())} pairs, max |dt*| = {d_res:.3e} "
              f"(tol {tol_res}); unresolved-class agreement: {ok_class}")
        if not ok_class or d_res > tol_res:
            fails.append("stored benchmark t* values not reproduced in the resolved zone")
    else:
        print("  [5] benchmark npz absent -- cross-check skipped")

    return fails


def t_cross_or_nan(rh, rc, v, n):
    TP = np.linspace(0.0, 200.0, 400)
    DECAY = np.exp(np.outer(v.evals, TP))
    DEN = np.sum(v.left_vecs.conj() * v.right_vecs, axis=0)
    LOGN = np.log(n)

    def d_curve(rv):
        c = (v.left_vecs.conj().T @ rv) / DEN
        M = (v.right_vecs @ (c[:, None] * DECAY)).T.reshape(len(TP), n, n)
        ev = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            return LOGN + np.where(ev > 1e-15, ev * np.log(ev), 0.0).sum(1)

    dh, dc = d_curve(rh), d_curve(rc)
    if dh[0] <= dc[0]:
        return np.nan
    diff = dc - dh
    idx = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return float(TP[i] + (TP[i + 1] - TP[i]) * (-diff[i]) / (diff[i + 1] - diff[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    args = ap.parse_args()

    z = np.load(GRAPH)
    print(f"canonical graph: {GRAPH.name}  ({z['tau'].shape[0]} nodes, "
          f"{int(z['tau'].sum()) // 2} edges, p={float(z['p'])})")
    print("rebuilding (dense eig 1296x1296)...")
    v = rebuild(z)

    print("verifying reconstruction against canonical fingerprints:")
    fails = verify(v, z)
    if fails:
        print("\nFAILED -- checkpoint NOT written:")
        for f in fails:
            print("  -", f)
        sys.exit(1)

    if args.out.exists() and not args.force:
        print(f"\nall checks passed; {args.out.name} already exists -- use --force to overwrite")
        return
    joblib.dump(v, args.out, compress=0)
    print(f"\nall checks passed -> wrote {args.out}")


if __name__ == "__main__":
    main()
