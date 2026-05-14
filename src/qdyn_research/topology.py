import random
from collections import deque

import numpy as np


def check_connectivity(tau: np.ndarray) -> bool:
    """Breadth-first connectivity test for the graph encoded by adjacency matrix tau."""
    n = tau.shape[0]
    if n == 0:
        return True
    visited = set([0])
    queue = deque([0])
    count = 0
    while queue:
        u = queue.popleft()
        count += 1
        neighbors = np.where(tau[u, :] == 1)[0]
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return count == n


def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Generate nearest-neighbor 2D grid adjacency matrix."""
    n = height * width
    tau = np.zeros((n, n), dtype=int)
    if height == 1 or width == 1:
        for i in range(n - 1):
            tau[i, i + 1] = 1
            tau[i + 1, i] = 1
        return tau
    for i in range(n):
        if (i + 1) % width != 0:
            tau[i, i + 1] = 1
            tau[i + 1, i] = 1
        if i < n - width:
            tau[i, i + width] = 1
            tau[i + width, i] = 1
    return tau


def generate_grid_with_manual_links_tau(height: int, width: int, extra_links: list[tuple[int, int]]) -> np.ndarray:
    """Grid adjacency with user-specified additional undirected links."""
    tau = generate_grid_tau(height, width)
    n = height * width
    for u, v in extra_links:
        if 0 <= u < n and 0 <= v < n:
            tau[u, v] = 1
            tau[v, u] = 1
    return tau


def generate_rewired_grid_tau(height: int, width: int, p: float, seed: int | None = None) -> np.ndarray:
    """Single-pass Watts-Strogatz-style rewiring with edge-level probability p.

    Does **not** guarantee connectivity of the resulting graph. Suitable for
    topological statistics (e.g. average path length via the largest connected
    component) where disconnected samples are handled explicitly by the caller.

    Algorithm differences vs. the connectivity-guaranteed variant:
    - No rejection loop: each edge is rewired at most once.
    - ``forbidden_nodes`` does *not* exclude the original partner, so a rewired
      edge may accidentally reconnect to its previous endpoint (harmless for
      statistical averages, but avoided in the guaranteed variant).

    Args:
        height: Number of rows in the base grid.
        width:  Number of columns in the base grid.
        p:      Per-edge rewiring probability in [0, 1].
        seed:   Optional integer seed for :mod:`random` PRNG.  When given, the
                module-level state is set via ``random.seed(seed)`` before any
                random calls so that results are exactly reproducible.
    """
    if seed is not None:
        random.seed(seed)
    initial_tau = generate_grid_tau(height, width)
    tau = initial_tau.copy()
    n = height * width
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if initial_tau[i, j] == 1]
    random.shuffle(edges)
    for u, v in edges:
        if random.random() < p:
            node_to_rewire = u if random.random() < 0.5 else v
            neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
            forbidden_nodes = set(neighbors) | {node_to_rewire}
            valid_targets = [node for node in range(n) if node not in forbidden_nodes]
            if valid_targets:
                w = random.choice(valid_targets)
                tau[u, v] = 0
                tau[v, u] = 0
                tau[node_to_rewire, w] = 1
                tau[w, node_to_rewire] = 1
    return tau


def generate_rewired_grid_tau_guaranteed_connectivity(
    height: int, width: int, p: float, seed: int | None = None
) -> np.ndarray:
    """Rewire edges with probability p; resample until the graph is connected.

    Suitable for spectral analysis of the Lindblad/Liouvillian operator, which
    requires a **unique** stationary state. A disconnected graph would produce
    multiple zero eigenvalues and invalidate the spectral decomposition.

    Algorithm differences vs. :func:`generate_rewired_grid_tau`:
    - Rejection loop: the entire rewiring is redrawn whenever the resulting
      graph is disconnected.
    - ``forbidden_nodes`` additionally excludes ``original_partner`` of the
      rewired endpoint, preventing the creation of trivial multi-edges back to
      the original neighbour.

    Args:
        height: Number of rows in the base grid.
        width:  Number of columns in the base grid.
        p:      Per-edge rewiring probability in [0, 1].
        seed:   Optional integer seed for :mod:`random` PRNG.  When given, the
                module-level state is set via ``random.seed(seed)`` **once**
                before the first rewiring attempt.  Subsequent rejection-loop
                iterations continue from the evolved PRNG state so that the
                seed uniquely determines the returned graph.
    """
    if seed is not None:
        random.seed(seed)
    while True:
        initial_tau = generate_grid_tau(height, width)
        tau = initial_tau.copy()
        n = height * width
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if initial_tau[i, j] == 1]
        random.shuffle(edges)

        for u, v in edges:
            if random.random() < p:
                node_to_rewire = u if random.random() < 0.5 else v
                original_partner = v if node_to_rewire == u else u
                neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
                forbidden_nodes = set(neighbors) | {node_to_rewire, original_partner}
                valid_targets = [node for node in range(n) if node not in forbidden_nodes]
                if valid_targets:
                    w = random.choice(valid_targets)
                    tau[u, v] = 0
                    tau[v, u] = 0
                    tau[node_to_rewire, w] = 1
                    tau[w, node_to_rewire] = 1

        if check_connectivity(tau):
            return tau

