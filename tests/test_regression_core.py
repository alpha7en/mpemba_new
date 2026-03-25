import numpy as np

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map, entropic_distance
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import generate_grid_tau, generate_rewired_grid_tau_guaranteed_connectivity


def test_diagonal_dephasing_entries_are_zero_for_populations():
    tau = generate_grid_tau(3, 3)
    l = build_liouvillian_dense(tau, J=1.0, gamma=0.1)
    n = tau.shape[0]
    diagonal = np.diag(l)
    diag_indices = np.arange(n) * (n + 1)
    assert np.allclose(diagonal[diag_indices].real, 0.0, atol=1e-12)


def test_excitability_map_is_non_negative():
    tau = generate_grid_tau(2, 2)
    l = build_liouvillian_dense(tau, J=1.0, gamma=0.1)
    _, left, right = analyze_liouvillian_modes_dense_strict(l)
    b_map = calculate_excitability_map(left, right, 1, 4)
    assert np.all(b_map >= -1e-14)


def test_entropy_distance_order_difference_is_explicit():
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 0.5
    rho[1, 1] = 0.5
    vec_c = rho.flatten(order="C")
    vec_f = rho.flatten(order="F")
    # For diagonal rho, both orders coincide; the test guards accidental changes in API behavior.
    assert np.isclose(entropic_distance(vec_c, 4, reshape_order="C"), entropic_distance(vec_f, 4, reshape_order="F"))


def test_rewired_generator_keeps_connectivity():
    tau = generate_rewired_grid_tau_guaranteed_connectivity(4, 4, 0.3)
    visited = set([0])
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in np.where(tau[u] == 1)[0]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    assert len(visited) == tau.shape[0]

