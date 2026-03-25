import numpy as np

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map, calculate_ipr, get_projection_coefficient
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import check_connectivity, generate_grid_tau


def test_grid_connectivity_and_liouvillian_shapes():
    tau = generate_grid_tau(3, 3)
    assert tau.shape == (9, 9)
    assert check_connectivity(tau)

    L = build_liouvillian_dense(tau, J=1.0, gamma=0.1)
    assert L.shape == (81, 81)


def test_projection_and_ipr_are_finite():
    tau = generate_grid_tau(2, 2)
    L = build_liouvillian_dense(tau, J=1.0, gamma=0.1)
    vals, left, right = analyze_liouvillian_modes_dense_strict(L)

    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1.0
    coeff = get_projection_coefficient(left[:, 1], right[:, 1], state)
    assert np.isfinite(coeff.real)

    b_map = calculate_excitability_map(left, right, 1, 4)
    assert b_map.shape == (4,)
    assert np.all(np.isfinite(b_map))

    ipr = calculate_ipr(right[:, 1], n=4, reshape_order="F")
    assert np.isfinite(ipr)

