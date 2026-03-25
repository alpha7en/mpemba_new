"""Topology generation utilities for rewired 2D lattices.

The routines in this module preserve the original script logic:
1) build a nearest-neighbor rectangular grid;
2) perform Watts-Strogatz-like edge rewiring;
3) reject rewiring attempts that break global connectivity.
"""

from __future__ import annotations

from collections import deque
import random

import numpy as np


def check_connectivity(tau: np.ndarray) -> bool:
    """Return True when the undirected graph encoded by `tau` is connected."""
    node_count = tau.shape[0]
    if node_count == 0:
        return True

    visited = {0}
    queue = deque([0])
    seen_nodes = 0

    while queue:
        node = queue.popleft()
        seen_nodes += 1
        neighbors = np.where(tau[node, :] == 1)[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return seen_nodes == node_count


def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Construct the base symmetric adjacency matrix for an `height x width` grid."""
    total_nodes = height * width
    tau = np.zeros((total_nodes, total_nodes), dtype=int)

    for node in range(total_nodes):
        if (node + 1) % width != 0:
            tau[node, node + 1] = 1
            tau[node + 1, node] = 1
        if node < total_nodes - width:
            tau[node, node + width] = 1
            tau[node + width, node] = 1

    return tau


def generate_rewired_grid_tau_guaranteed_connectivity(height: int, width: int, p: float) -> np.ndarray:
    """Generate rewired grid topology while strictly enforcing connectivity.

    The implementation mirrors the original production script used for expensive
    spectrum generation, including endpoint selection and candidate filtering.
    """
    while True:
        initial_tau = generate_grid_tau(height, width)
        tau = initial_tau.copy()
        total_nodes = height * width
        base_edges = [(i, j) for i in range(total_nodes) for j in range(i + 1, total_nodes) if initial_tau[i, j] == 1]
        random.shuffle(base_edges)

        for u, v in base_edges:
            if random.random() < p:
                node_to_rewire = u if random.random() < 0.5 else v
                original_partner = v if node_to_rewire == u else u

                neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
                forbidden_nodes = set(neighbors) | {node_to_rewire, original_partner}
                valid_targets = [n for n in range(total_nodes) if n not in forbidden_nodes]

                if valid_targets:
                    w = random.choice(valid_targets)
                    tau[u, v] = 0
                    tau[v, u] = 0
                    tau[node_to_rewire, w] = 1
                    tau[w, node_to_rewire] = 1

        if check_connectivity(tau):
            return tau

