"""CF-2-corrected spectral sweep: RIGHTMOST (max Re) modes vs rewiring probability p.

Replaces the legacy sweep (run_multicore_npz.py), which ordered modes by |lambda| and
provably missed the slowest oscillating multiplets (see docs/revision_log.md, Block B1,
and scripts/verify_rightmost_solver.py for the dense-truth verification of the solver).

Design differences vs the legacy driver:
- run_single_sparse_rewiring_job_rightmost: modes ordered by Re(lambda) via the exponential
  transform; per-mode residuals stored in the archive as a quality trail.
- CRASH-SAFE: one part-file per p value (precalc/sweep_rightmost_parts/part_XX.npz), written
  as soon as that p completes; re-running skips finished parts (resume after interruption).
- Final merge into precalc/sweep_rightmost_10x10.npz. The name deliberately does NOT match
  the fig6-8 glob patterns: switching the figures to the corrected archive is an explicit
  later step (Block B figure redesign), not a silent side effect.

Same ensemble grid as the legacy sweep (comparability): p = logspace(-4, 0, 40), 30 seeded
runs per p (seed = run_idx, identical topologies to the legacy archive), J=1, gamma=0.1.
"""
import os

# BLAS pinning BEFORE numpy import (see run_multicore_npz.py for the rationale)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import concurrent.futures
import time
from pathlib import Path

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.npz_io import run_single_sparse_rewiring_job_rightmost

HEIGHT = WIDTH = 10
J, GAMMA = 1.0, 0.1
NUM_MODES = 10                    # stationary + ~2 full quadruplets + margin
P_VALUES = np.logspace(-4, 0, num=40)
RUNS_PER_P = 30
WORKERS = max(1, (os.cpu_count() or 1) // 2)

PRECALC = Path(__file__).resolve().parent / "precalc"
PARTS = PRECALC / "sweep_rightmost_parts"
OUT = PRECALC / "sweep_rightmost_10x10.npz"


def compute_one_p(pi, p):
    """All runs for one p value -> dict of archive keys (empty entries dropped)."""
    data = {}
    n_fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_single_sparse_rewiring_job_rightmost,
                          p, run_idx, HEIGHT, WIDTH, J, GAMMA, NUM_MODES)
                for run_idx in range(RUNS_PER_P)]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r:
                data.update(r)
            else:
                n_fail += 1
    return data, n_fail


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_fail = 0
    for pi, p in enumerate(P_VALUES):
        part = PARTS / f"part_{pi:02d}.npz"
        if part.exists():
            print(f"[{pi+1:02d}/40] p={p:.5f}  SKIP (part exists)", flush=True)
            continue
        data, n_fail = compute_one_p(pi, p)
        total_fail += n_fail
        np.savez_compressed(part, **data)
        done = pi + 1
        eta = (time.time() - t0) / max(1, done) * (40 - done)
        print(f"[{done:02d}/40] p={p:.5f}  runs={RUNS_PER_P - n_fail}/{RUNS_PER_P}"
              f"  ({time.time()-t0:.0f}s, ETA {eta/60:.0f} min)", flush=True)

    # merge parts
    merged = {}
    worst_resid = 0.0
    n_runs = 0
    for pi in range(len(P_VALUES)):
        part = PARTS / f"part_{pi:02d}.npz"
        if not part.exists():
            continue
        z = np.load(part)
        for k in z.files:
            merged[k] = z[k]
            if k.endswith("_residuals"):
                worst_resid = max(worst_resid, float(z[k].max()))
                n_runs += 1
    np.savez_compressed(OUT, **merged)
    print(f"\nmerged {n_runs} runs -> {OUT.name}  (failed jobs: {total_fail}, "
          f"worst mode residual: {worst_resid:.2e})", flush=True)


if __name__ == "__main__":
    main()
