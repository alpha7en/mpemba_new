"""Eigenmode analysis for non-Hermitian Liouvillian operators."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu


def analyze_liouvillian_modes_dense(liouvillian: np.ndarray):
    """Return sorted eigenvalues and biorthogonal vectors for dense Liouvillian."""
    eigenvalues, left_vectors, right_vectors = eig(liouvillian, left=True, right=True)
    sort_indices = np.argsort(eigenvalues.real)[::-1]
    return (
        eigenvalues[sort_indices],
        left_vectors[:, sort_indices],
        right_vectors[:, sort_indices],
    )


def analyze_liouvillian_modes_sparse_robust(liouvillian: csc_matrix, num_modes: int):
    """Compute slow sparse modes using shift-invert, matching legacy script behavior.

    Sparse ARPACK runs may fail for some random rewired samples; for compatibility
    with the original multicore pipeline those failures are represented as
    `(None, None)` and filtered by the caller.
    """
    sigma = 1e-9 + 0j
    try:
        lu = splu(liouvillian - sigma * identity(liouvillian.shape[0], dtype=np.complex128, format="csc"))
        op_inv = LinearOperator(liouvillian.shape, matvec=lu.solve, dtype=liouvillian.dtype)
        mu_values, mu_vectors = eigs(op_inv, k=num_modes, which="LM")
        lambda_values = 1.0 / mu_values + sigma
        sort_indices = np.argsort(lambda_values.real)[::-1]
        return lambda_values[sort_indices], mu_vectors[:, sort_indices]
    except (RuntimeError, ArpackNoConvergence, ValueError):
        return None, None
