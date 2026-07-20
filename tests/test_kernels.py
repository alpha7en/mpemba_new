"""Fast smoke tests for the computational kernels (topology, Liouvillian, projections, IPR).

These make good on the guardrails described in docs/equivalence_notes.md: every test either
checks an exact mathematical identity or a convention the paper relies on. Runtime ~1 min.
"""
import numpy as np
import pytest

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import (
    calculate_ipr,
    entropic_distance,
    factor_mode_basis,
    project_rho_on_modes_exact,
)
from qdyn_research.topology import (
    generate_grid_tau,
    generate_rewired_grid_tau_guaranteed_connectivity,
)


# ---------------------------------------------------------------- topology
def test_regular_grid_edge_count():
    tau = generate_grid_tau(3, 3)
    assert tau.shape == (9, 9)
    assert np.array_equal(tau, tau.T)
    assert tau.sum() // 2 == 12          # h*(w-1) + w*(h-1) = 6 + 6


def test_rewired_topology_seed_reproducible_and_connected():
    import networkx as nx

    a = generate_rewired_grid_tau_guaranteed_connectivity(4, 4, 0.2, seed=42)
    b = generate_rewired_grid_tau_guaranteed_connectivity(4, 4, 0.2, seed=42)
    c = generate_rewired_grid_tau_guaranteed_connectivity(4, 4, 0.2, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert nx.is_connected(nx.from_numpy_array(a))


# ---------------------------------------------------------------- Liouvillian
def test_liouvillian_dimensions():
    tau = generate_grid_tau(2, 3)
    L = build_liouvillian_dense(tau, 1.0, 0.1)
    assert L.shape == (36, 36)           # N^2 x N^2 for N = 6 nodes


def test_liouvillian_matches_physical_lindbladian():
    """Code generator at parameter 1.0 == Lindbladian with H = -(1/2) tau (vec-F column
    stacking). This is the J-convention statement of the paper (J = 1/2), verified exactly."""
    tau = generate_grid_tau(2, 3).astype(float)
    N, gamma = 6, 0.1
    L_code = build_liouvillian_dense(tau, 1.0, gamma)
    H = -0.5 * tau
    I = np.eye(N)
    coh = -1j * (np.kron(I, H) - np.kron(H.T, I))
    deph = np.zeros((N * N, N * N), complex)
    for a in range(N):
        for b in range(N):
            if a != b:
                deph[a * N + b, a * N + b] = -gamma
    assert np.abs(L_code - (coh + deph)).max() == 0.0


def test_dephasing_acts_only_on_coherences():
    """With J=0 the generator is diagonal: 0 on populations, -gamma on every coherence."""
    tau = generate_grid_tau(2, 2).astype(float)
    N, gamma = 4, 0.37
    L = build_liouvillian_dense(tau, 0.0, gamma)
    assert np.abs(L - np.diag(np.diag(L))).max() == 0.0
    d = np.diag(L)
    for a in range(N):
        for b in range(N):
            expected = 0.0 if a == b else -gamma
            assert abs(d[a * N + b] - expected) < 1e-15


# ---------------------------------------------------------------- metrics
def test_entropic_distance_limits():
    n = 5
    uniform = (np.eye(n) / n).astype(complex).flatten()
    pure = np.zeros((n, n), complex)
    pure[0, 0] = 1.0
    assert abs(entropic_distance(uniform, n, "C")) < 1e-12
    assert abs(entropic_distance(pure.flatten(), n, "C") - np.log(n)) < 1e-12


def test_ipr_limits():
    n = 9
    uniform = np.diag(np.full(n, 1.0 / n)).astype(complex).flatten(order="F")
    single = np.zeros((n, n), complex)
    single[2, 2] = 1.0
    assert abs(calculate_ipr(uniform, n, "F") - 1.0 / n) < 1e-12
    assert abs(calculate_ipr(single.flatten(order="F"), n, "F") - 1.0) < 1e-12


def test_dirichlet_diag_state_is_valid_density_matrix():
    from qdyn_research.mpemba_validation import MpembaValidator

    np.random.seed(0)
    v = MpembaValidator(height=3, width=3, p=0.1, seed=1)
    rho = v.generate_random_diag_density_matrix().reshape(9, 9, order="C")
    off = rho - np.diag(np.diag(rho))
    assert np.abs(off).max() == 0.0
    assert abs(np.trace(rho).real - 1.0) < 1e-12
    assert np.all(np.diag(rho).real >= 0)


# ---------------------------------------------------------------- projections
def test_exact_projection_reconstructs_state():
    """V c = vec(rho) must reconstruct rho to ~1e-10 even on a degenerate regular lattice
    (the pairwise biorthogonal division does NOT, which was referee finding CF-1)."""
    tau = generate_grid_tau(4, 4)
    L = build_liouvillian_dense(tau, 1.0, 0.1)
    evals, right = np.linalg.eig(L)
    lu = factor_mode_basis(right)
    rho = np.zeros((16, 16), complex)
    rho[5, 5] = 0.7
    rho[10, 10] = 0.3
    _c, residual = project_rho_on_modes_exact(rho, lu, right, order="F")
    assert residual < 1e-10


def test_group_diag_weight_basis_invariant():
    """The per-group diagonal weight used in Figs. 7-8 must not change under a unitary
    rotation of the vectors spanning the same (degenerate) subspace."""
    from reduce_sweep_groups import group_diag_weight

    rng = np.random.default_rng(3)
    raw = rng.standard_normal((10000, 2)) + 1j * rng.standard_normal((10000, 2))
    q, _ = np.linalg.qr(raw)
    th = 0.7
    u = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], complex)
    d1 = group_diag_weight(q)
    d2 = group_diag_weight(q @ u)
    assert np.abs(d1 - d2).max() < 1e-12
