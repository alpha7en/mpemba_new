"""Connectivity statistics of the single-pass (non-rejection) rewiring used for Fig. 4.

Fig. 4 (L_avg vs p) uses `generate_rewired_grid_tau`, which does NOT enforce connectivity;
for disconnected realizations `calculate_average_shortest_path_length` falls back to the
largest connected component (LCC) — the convention stated in the paper (Sec. 4).

This script quantifies how often that fallback actually fires and how large the LCC is,
as backing data for a potential reviewer question. It is NOT cited in the paper text.

Reproducibility: the topology generator draws from the module-level `random` PRNG, which is
seeded once below, so the ensemble is deterministic.
"""
import random

import networkx as nx
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import generate_rewired_grid_tau

SEED = 1
HEIGHT = WIDTH = 10
N = HEIGHT * WIDTH
RUNS_PER_P = 1000
P_VALUES = [1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0]


def main():
    random.seed(SEED)
    print(f"{HEIGHT}x{WIDTH} grid, {RUNS_PER_P} realizations per p, seed={SEED}")
    print(f"{'p':>8}  {'disconnected':>12}  {'mean LCC':>9}  {'min LCC':>8}")
    for p in P_VALUES:
        n_disc = 0
        lcc_sizes = []
        for _ in range(RUNS_PER_P):
            g = nx.from_numpy_array(generate_rewired_grid_tau(HEIGHT, WIDTH, p))
            if nx.is_connected(g):
                lcc_sizes.append(N)
            else:
                n_disc += 1
                lcc_sizes.append(len(max(nx.connected_components(g), key=len)))
        lcc_sizes = np.array(lcc_sizes)
        print(f"{p:>8g}  {100 * n_disc / RUNS_PER_P:>11.1f}%  {lcc_sizes.mean():>9.1f}  {lcc_sizes.min():>8d}")


if __name__ == "__main__":
    main()
