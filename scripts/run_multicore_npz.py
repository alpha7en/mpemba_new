import concurrent.futures
import os
import time

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.npz_io import run_single_sparse_rewiring_job, save_npz_bundle


"""Batch spectral precomputation for the rewired-grid Lindblad model.

Workflow for each sample (p, run_idx):
1) generate connected rewired adjacency matrix tau,
2) build sparse Liouvillian L = -i(I⊗H - H^T⊗I) + L_D with H = -(J/2)tau,
3) extract slow modes near Re(lambda)≈0 using shift-invert,
4) save eigenvalues/eigenvectors to NPZ for fast post-processing plots.
"""

def main():
    # ---------------------------
    # System parameters
    # ---------------------------
    height = 10
    width = 10
    j = 1.0
    gamma = 0.1
    num_modes_to_find = 4

    # ---------------------------
    # Ensemble sampling parameters
    # ---------------------------
    # p grid is logarithmic to resolve small-rewiring regime where spectral changes are sharp.
    p_values = np.logspace(-4, 0, num=40)
    runs_per_p = 30

    # Total number of independent graph realizations.
    total = len(p_values) * runs_per_p
    completed = 0
    data_to_save = {}

    # Conservative parallelism for memory-intensive sparse eigensolves.
    workers = max(1, (os.cpu_count() or 1) // 2)
    started = time.time()

    # Each task writes keys:
    #   p_{p}_run_{idx}_lambdas, p_{p}_run_{idx}_vectors, p_{p}_run_{idx}_p_value
    # so the archive can be parsed directly by fig6/fig7 statistics scripts.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for p in p_values:
            for run_idx in range(runs_per_p):
                futures.append(
                    executor.submit(
                        run_single_sparse_rewiring_job,
                        p,
                        run_idx,
                        height,
                        width,
                        j,
                        gamma,
                        num_modes_to_find,
                    )
                )

        for future in concurrent.futures.as_completed(futures):
            data_to_save.update(future.result())
            completed += 1
            elapsed = time.time() - started
            avg = elapsed / completed
            remaining = avg * (total - completed)
            print(f"Progress: {completed}/{total} ({100 * completed / total:.2f}%), ETA {remaining / 60:.2f} min")

    if data_to_save:
        # A single compressed file avoids repeating multi-hour spectral runs.
        out_file = save_npz_bundle(data_to_save, height=height, width=width)
        print(f"Saved {len(data_to_save) // 3} runs to {out_file}")
    else:
        print("No data produced.")


if __name__ == "__main__":
    main()

