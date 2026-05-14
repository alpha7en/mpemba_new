import datetime
import os
from collections import defaultdict

import numpy as np

from .liouvillian import build_liouvillian_sparse
from .spectral import analyze_liouvillian_modes_sparse_robust
from .topology import generate_rewired_grid_tau_guaranteed_connectivity


def run_single_sparse_rewiring_job(p, run_idx, height, width, j, gamma, num_modes):
    """Run one sparse spectral sample and store keys p_{p}_run_{idx}_{lambdas,vectors,p_value,tau}.

    The adjacency matrix tau is generated with seed=run_idx so that the same
    graph can be exactly reproduced in analysis scripts (e.g. fig8_overlap.py)
    by calling generate_rewired_grid_tau_guaranteed_connectivity with the same
    seed.  tau is also stored in the NPZ bundle as a direct source of truth.
    """
    # seed = run_idx guarantees deterministic topology per (p, run_idx) pair.
    tau = generate_rewired_grid_tau_guaranteed_connectivity(height, width, p, seed=run_idx)
    liouvillian = build_liouvillian_sparse(tau, j, gamma)
    lambdas, vectors = analyze_liouvillian_modes_sparse_robust(liouvillian, num_modes=num_modes)
    if lambdas is None:
        return {}
    prefix = f"p_{p}_run_{run_idx}"
    return {
        f"{prefix}_lambdas": lambdas,
        f"{prefix}_vectors": vectors,
        f"{prefix}_p_value": p,
        f"{prefix}_tau": tau,  # adjacency matrix saved for exact graph reproduction in fig8
    }


def save_npz_bundle(data: dict, height: int, width: int, out_dir: str = ".") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rewiring_spectrum_data_{height}x{width}_{stamp}.npz"
    path = os.path.join(out_dir, filename)
    np.savez_compressed(path, **data)
    return path


def parse_grouped_lambdas(npz_file: str):
    data = np.load(npz_file)
    grouped = defaultdict(list)
    for key in data.keys():
        if key.endswith("_lambdas"):
            try:
                p = float(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            grouped[p].append(data[key])
    return grouped


def parse_grouped_vectors(npz_file: str):
    data = np.load(npz_file)
    grouped = defaultdict(list)
    for key in data.keys():
        if key.endswith("_vectors"):
            base = key.replace("_vectors", "")
            p_key = f"{base}_p_value"
            if p_key in data:
                grouped[data[p_key].item()].append(data[key])
    return grouped

