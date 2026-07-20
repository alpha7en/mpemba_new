"""Advantage control for the Sec. 8.4 benchmark (referee item CF-3).

The crossing of the engineered pair occurs at D ~ 3.6e-3 (both trajectories already close to
equilibrium). To quantify how strong the resulting Mpemba effect is BEYOND the mere existence
of a crossing, we compute for every admissible random pair the maximum subsequent advantage

    dD_max = max_t [ D_cold(t) - D_hot(t) ]   (>0 only if the pair actually crosses),

i.e. by how much the initially-farther ("hot") state overtakes the initially-closer one, and
compare the engineered pair's advantage against this ensemble.

The random ensemble is IDENTICAL to run_dirichlet_benchmark.py (same SEED=7, same batching,
same gap filter), so the two files describe the same 30 000 admissible pairs. Output:
precalc/dirichlet_advantage_30k.npz (advantages, adv_algo, percentile).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from pathlib import Path

import joblib
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.metrics import calculate_excitability_map, entropic_distance
from qdyn_research.mpemba import find_guaranteed_mpemba_dense

PRECALC = Path(__file__).resolve().parent / "precalc"
CHECKPOINT = PRECALC / "validator_6x6_p015.pkl"
OUT = PRECALC / "dirichlet_advantage_30k.npz"

SEED = 7
N_ADM_TARGET = 30000
BATCH = 40000
T_WINDOW = np.linspace(0.0, 200.0, 400)
N_JOBS = 6

V = joblib.load(CHECKPOINT)
V.n = int(V.N)
n = V.n
LV, RV, evals = V.left_vecs, V.right_vecs, V.evals
DEN = np.sum(LV.conj() * RV, axis=0)
LOGN = np.log(n)
DECAY = np.exp(np.outer(evals, T_WINDOW))


def d_curve(rho_vec):
    c = (LV.conj().T @ rho_vec) / DEN
    M = (RV @ (c[:, None] * DECAY)).T.reshape(len(T_WINDOW), n, n)
    ev = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(ev > 1e-15, ev * np.log(ev), 0.0).sum(1)
    return LOGN + s


def pair_advantage(pop_hot, pop_cold):
    dh = d_curve(np.diag(pop_hot).astype(complex).flatten("C"))
    dc = d_curve(np.diag(pop_cold).astype(complex).flatten("C"))
    return float(max(0.0, np.max(dc - dh)))


def main():
    b_map = calculate_excitability_map(LV, RV, 1, n)
    vec_hot, _vc, vec_cold = find_guaranteed_mpemba_dense(
        LV, RV, n, excitability_map=b_map, mode_idx=1, distance_order="C"
    )
    adv_algo = float(max(0.0, np.max(d_curve(vec_cold) - d_curve(vec_hot))))
    gap_algo = float(entropic_distance(vec_hot, n, "C") - entropic_distance(vec_cold, n, "C"))
    gap_min = min(LOGN, gap_algo) / 10.0
    print(f"engineered pair: dD_max = {adv_algo:.6f}  (gap_min filter {gap_min:.3f})", flush=True)

    rng = np.random.default_rng(SEED)
    advantages = []
    t0 = time.time()
    while len(advantages) < N_ADM_TARGET:
        P = rng.dirichlet(np.ones(n), size=BATCH)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = LOGN + np.where(P > 1e-15, P * np.log(P), 0.0).sum(1)
        pa, pb, Da, Db = P[0::2], P[1::2], D[0::2], D[1::2]
        swap = Da < Db
        hot = np.where(swap[:, None], pb, pa)
        cold = np.where(swap[:, None], pa, pb)
        keep = (np.where(swap, Db, Da) - np.where(swap, Da, Db)) >= gap_min
        hot, cold = hot[keep], cold[keep]
        res = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(pair_advantage)(hot[k], cold[k]) for k in range(len(hot))
        )
        advantages += list(res)
        print(f"  admissible {len(advantages)}  ({time.time() - t0:.0f}s)", flush=True)

    adv = np.array(advantages[:N_ADM_TARGET])
    frac_better = float(np.mean(adv > adv_algo))
    np.savez_compressed(OUT, advantages=adv, adv_algo=adv_algo, frac_better=frac_better)
    print(f"\nadmissible pairs: {len(adv)}")
    print(f"pairs with larger advantage than engineered: {int((adv > adv_algo).sum())} "
          f"-> {100 * frac_better:.3f}%  (engineered percentile: {100 * (1 - frac_better):.2f})")
    print(f"advantage distribution: median={np.median(adv):.2e}  p90={np.percentile(adv, 90):.2e} "
          f"p99={np.percentile(adv, 99):.2e}  max={adv.max():.2e}")
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
