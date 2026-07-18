"""Monte-Carlo benchmark of the deterministic Mpemba-pair algorithm (Sec. 8.4, Fig. 13 data).

Generates `precalc/dirichlet_benchmark_30k.npz`: crossing times t* for an ensemble of admissible
random pairs of diagonal states, populations ~ Dirichlet(1,...,1), on the exact 6x6 p=0.15 graph
of the paper (precalc/validator_6x6_p015.pkl). A pair is admissible when its initial
relative-entropy gap is >= 10% of the gap of the engineered pair. t* is NaN when the pair does
not cross within the [0, 200] window.

Reproducibility: all randomness comes from `np.random.default_rng(SEED)` in the main process
(workers only evaluate already-drawn pairs), so the output is bit-reproducible. SEED = 7 is the
run stored in the repository and quoted in the paper (1.21% of admissible pairs cross faster
than the engineered pair, t*_algo = 44.66).

The D(t) evaluation below is a vectorized equivalent of the spectral kernel
(worker_get_decomposition + entropic_distance): all time points in one matmul, then a batched
eigvalsh. It was verified to match the kernel to machine precision (max |dD| ~ 3.6e-15).
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
OUT = PRECALC / "dirichlet_benchmark_30k.npz"

SEED = 7                # rng seed of the stored run — do not change when regenerating the artifact
N_ADM_TARGET = 30000    # admissible pairs collected
BATCH = 40000           # Dirichlet states drawn per batch (BATCH/2 candidate pairs)
T_WINDOW = np.linspace(0.0, 200.0, 400)   # res 0.5 + linear interpolation at the sign change
N_JOBS = 6

V = joblib.load(CHECKPOINT)
V.n = int(V.N)
n = V.n
LV, RV, evals = V.left_vecs, V.right_vecs, V.evals
DEN = np.sum(LV.conj() * RV, axis=0)      # biorthogonal norms <w_k|v_k>
LOGN = np.log(n)
tau_relax = 1.0 / abs(np.sort(evals.real)[::-1][1])
DECAY = np.exp(np.outer(evals, T_WINDOW))


def t_cross(rho_hot, rho_cold):
    """First time where D_cold(t) overtakes D_hot(t) (linear-interpolated); NaN if none."""

    def d_curve(rho_vec):
        c = (LV.conj().T @ rho_vec) / DEN
        M = (RV @ (c[:, None] * DECAY)).T.reshape(len(T_WINDOW), n, n)   # order 'C'
        ev = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(ev > 1e-15, ev * np.log(ev), 0.0).sum(1)
        return LOGN + s

    d_hot, d_cold = d_curve(rho_hot), d_curve(rho_cold)
    if d_hot[0] <= d_cold[0]:
        return np.nan
    diff = d_cold - d_hot
    idx = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return float(T_WINDOW[i] + (T_WINDOW[i + 1] - T_WINDOW[i]) * (-diff[i]) / (diff[i + 1] - diff[i]))


def pair_t_cross(pop_hot, pop_cold):
    return t_cross(np.diag(pop_hot).astype(complex).flatten("C"),
                   np.diag(pop_cold).astype(complex).flatten("C"))


def main():
    # engineered pair on the same grid/graph (deterministic construction)
    b_map = calculate_excitability_map(LV, RV, 1, n)
    vec_hot, _vec_cold, vec_cold_score = find_guaranteed_mpemba_dense(
        LV, RV, n, excitability_map=b_map, mode_idx=1, distance_order="C"
    )
    t_algo = t_cross(vec_hot, vec_cold_score)
    gap_algo = float(entropic_distance(vec_hot, n, "C") - entropic_distance(vec_cold_score, n, "C"))
    gap_min = min(LOGN, gap_algo) / 10.0
    print(f"t_algo={t_algo:.3f}  tau={tau_relax:.3f}  gap_min={gap_min:.3f}", flush=True)

    # random ensemble: draws in the MAIN process only (bit-reproducible), workers just evaluate
    rng = np.random.default_rng(SEED)
    tstars = []
    t0 = time.time()
    while len(tstars) < N_ADM_TARGET:
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
            joblib.delayed(pair_t_cross)(hot[k], cold[k]) for k in range(len(hot))
        )
        tstars += list(res)
        print(f"  admissible {len(tstars)}  ({time.time() - t0:.0f}s)", flush=True)

    ts = np.array(tstars[:N_ADM_TARGET])
    n_adm = len(ts)
    faster = int(np.nansum(ts < t_algo))
    np.savez(OUT, tstars=ts, t_algo=t_algo, tau=tau_relax, gap_algo=gap_algo, n_adm=n_adm)
    print(f"\nadmissible pairs: {n_adm}")
    print(f"cross faster than algo: {faster} -> {100 * faster / n_adm:.3f}% of admissible pairs")
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
