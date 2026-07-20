"""Consistency tests for the canonical data artifacts (paper graph, checkpoint, benchmark).

Tests that need the local 44-MB checkpoint are skipped automatically on a fresh clone
(rebuild it with scripts/rebuild_validator.py, which itself runs the full 5-check suite).
"""
from pathlib import Path

import numpy as np
import pytest

PRECALC = Path(__file__).resolve().parent.parent / "scripts" / "precalc"
GRAPH = PRECALC / "paper_graph_6x6_p015.npz"
PKL = PRECALC / "validator_6x6_p015.pkl"
BENCH = PRECALC / "dirichlet_benchmark_30k.npz"


def test_canonical_graph_fingerprints():
    z = np.load(GRAPH)
    tau = z["tau"]
    assert tau.shape == (36, 36)
    assert np.array_equal(tau, tau.T)
    assert int(tau.sum()) // 2 == 60
    assert int(z["hot_idx_expected"]) == 1
    assert z["cold_idx_expected"].tolist() == [17, 22, 23]
    assert abs(float(z["t_algo_expected"]) - 44.6600360539) < 1e-9
    assert abs(float(z["gap_algo_expected"]) - 1.0986122887) < 1e-9
    assert len(z["evals_expected"]) == 1296


@pytest.mark.skipif(not PKL.exists(), reason="local checkpoint absent (run rebuild_validator.py)")
def test_checkpoint_matches_canonical_graph():
    import joblib

    z = np.load(GRAPH)
    v = joblib.load(PKL)
    assert np.array_equal(v.tau, z["tau"])
    # spectra as sets (lexicographic complex sort is unstable for conjugate pairs)
    exp = z["evals_expected"]
    dist = np.abs(v.evals[:, None] - exp[None, :])
    hausdorff = max(dist.min(axis=1).max(), dist.min(axis=0).max())
    assert hausdorff < 1e-8


@pytest.mark.skipif(not (PKL.exists() and BENCH.exists()),
                    reason="checkpoint or benchmark data absent")
def test_benchmark_first_resolved_pairs_reproduce():
    """Regenerate the first admissible Dirichlet pairs (SEED=7) and compare the stored
    crossing times in the physically resolved zone (t* < 150; the noise tail is not
    reproducible across eigendecompositions by construction, see rebuild_validator.py)."""
    import run_dirichlet_benchmark as m

    d = np.load(BENCH)
    stored = d["tstars"][:6]

    rng = np.random.default_rng(m.SEED)
    P = rng.dirichlet(np.ones(m.n), size=m.BATCH)
    LOGN = np.log(m.n)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = LOGN + np.where(P > 1e-15, P * np.log(P), 0.0).sum(1)
    pa, pb, Da, Db = P[0::2], P[1::2], D[0::2], D[1::2]
    swap = Da < Db
    hot = np.where(swap[:, None], pb, pa)
    cold = np.where(swap[:, None], pa, pb)
    gap_min = min(LOGN, float(d["gap_algo"])) / 10.0
    keep = (np.where(swap, Db, Da) - np.where(swap, Da, Db)) >= gap_min
    hot, cold = hot[keep], cold[keep]

    for k in (3, 5):                       # stored resolved crossings: 95.18, 68.62
        t = m.pair_t_cross(hot[k], cold[k])
        assert abs(t - stored[k]) < 1e-4, f"pair {k}: {t} vs stored {stored[k]}"
