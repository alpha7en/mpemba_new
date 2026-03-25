"""Refactored entry-point for multicore spectrum generation.

This script preserves the original output key layout to remain compatible with
already produced `.npz` archives and downstream plotting scripts.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import os
import time

import numpy as np

from mpemba_refactored.eigensolver import analyze_liouvillian_modes_sparse_robust
from mpemba_refactored.liouvillian import build_liouvillian_sparse
from mpemba_refactored.npz_format import pack_run_result
from mpemba_refactored.topology import generate_rewired_grid_tau_guaranteed_connectivity


def single_run(p_value, run_idx, height, width, coupling_j, gamma, num_modes_to_find):
    tau = generate_rewired_grid_tau_guaranteed_connectivity(height, width, p_value)
    liouvillian = build_liouvillian_sparse(tau, coupling_j, gamma)
    lambdas, vectors = analyze_liouvillian_modes_sparse_robust(liouvillian, num_modes=num_modes_to_find)
    if lambdas is None:
        return {}
    return pack_run_result(p_value, run_idx, lambdas, vectors)


if __name__ == "__main__":
    HEIGHT = 10
    WIDTH = 10
    j = 1.0
    gamma = 0.1
    NUM_MODES_TO_FIND = 4

    p_values = np.logspace(-4, 0, num=40)
    num_runs_per_p = 30

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = f"rewiring_spectrum_data_{HEIGHT}x{WIDTH}_{timestamp}.npz"
    data_to_save = {}

    num_cores = os.cpu_count() or 1
    # Keep one half of logical cores to avoid overloading shared environments.
    max_workers = max(1, num_cores // 2)

    total_runs = len(p_values) * num_runs_per_p
    completed_runs = 0
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for p in p_values:
            for run_idx in range(num_runs_per_p):
                futures.append(executor.submit(single_run, p, run_idx, HEIGHT, WIDTH, j, gamma, NUM_MODES_TO_FIND))

        for future in concurrent.futures.as_completed(futures):
            data_to_save.update(future.result())
            completed_runs += 1
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / completed_runs
            remaining_runs = total_runs - completed_runs
            estimated_remaining = avg_time_per_run * remaining_runs
            percentage = (completed_runs / total_runs) * 100
            print(
                f"Progress: {completed_runs}/{total_runs} ({percentage:.2f}%), "
                f"ETA: {estimated_remaining / 60:.2f} min"
            )

    if data_to_save:
        np.savez_compressed(output_filename, **data_to_save)
        print(f"Saved: {os.path.abspath(output_filename)}")
    else:
        print("No data saved.")
