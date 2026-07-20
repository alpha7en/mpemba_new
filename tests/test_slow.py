"""Heavy verification tests (dense 1296x1296 eigendecompositions). Run with: pytest -m slow"""
from pathlib import Path

import numpy as np
import pytest

PRECALC = Path(__file__).resolve().parent.parent / "scripts" / "precalc"
GRAPH = PRECALC / "paper_graph_6x6_p015.npz"

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not GRAPH.exists(), reason="canonical graph artifact absent")
def test_rebuild_validator_passes_all_checks():
    """Full physics-identity suite: rebuild the checkpoint from the canonical graph and
    verify spectrum, engineered-pair selection, gap, t*, and the benchmark cross-check."""
    import rebuild_validator as rb

    z = np.load(rb.GRAPH)
    v = rb.rebuild(z)
    fails = rb.verify(v, z)
    assert fails == [], f"verification failures: {fails}"


def test_rightmost_solver_vs_dense_truth_6x6_regular():
    """The exp-transform Arnoldi solver must find the true rightmost (oscillating) modes
    of the regular 6x6 lattice -- the exact failure mode of the legacy sweep (CF-2)."""
    from qdyn_research.liouvillian import build_liouvillian_dense, build_liouvillian_sparse
    from qdyn_research.spectral import (
        analyze_liouvillian_modes_dense_strict,
        analyze_liouvillian_modes_rightmost,
    )
    from qdyn_research.topology import generate_grid_tau

    tau = generate_grid_tau(6, 6)
    evals, _l, _r = analyze_liouvillian_modes_dense_strict(build_liouvillian_dense(tau, 1.0, 0.1))
    truth = evals[np.argsort(evals.real)[::-1]]
    truth_nt = truth[np.abs(truth) > 1e-10]

    lam, _vecs, resid = analyze_liouvillian_modes_rightmost(
        build_liouvillian_sparse(tau, 1.0, 0.1), num_modes=20)
    assert lam is not None
    assert resid.max() < 1e-8
    lam_nt = lam[np.abs(lam) > 1e-10]

    # true rightmost mode found (it oscillates on the regular lattice)
    assert abs(truth_nt[0].imag) > 0.1
    assert np.abs(lam_nt - truth_nt[0]).min() < 1e-8
    # the top-4 distinct Re-groups are all represented
    groups = []
    for v in truth_nt[:60]:
        if not any(abs(v.real - g) < 1e-6 for g in groups):
            groups.append(v.real)
    for g in groups[:4]:
        assert np.min(np.abs(lam_nt.real - g)) < 1e-8
