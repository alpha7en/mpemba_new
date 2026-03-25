"""Liouvillian builders for dense and sparse workflows.

Vectorization convention is fixed to column-major (Fortran order) to preserve
the Kronecker-structured Lindblad mapping used throughout the research code.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, diags, identity, kron


def _dephasing_diagonal(size_n: int, gamma: float) -> np.ndarray:
    """Return diagonal entries for pure dephasing superoperator.

    For square matrices, diagonal density-matrix entries map to indices
    `i*(N+1)` in both C/F vectorization, and must have zero decay.
    """
    diag_ld = np.full(size_n * size_n, -gamma, dtype=np.complex128)
    diag_ld[np.arange(size_n) * (size_n + 1)] = 0.0
    return diag_ld


def build_liouvillian_dense(tau: np.ndarray, coupling_j: complex, gamma: float):
    """Build dense Hamiltonian and Liouvillian matrices from adjacency `tau`."""
    tau = np.asarray(tau, dtype=np.complex128)
    size_n = tau.shape[0]
    hamiltonian = -coupling_j / 2.0 * tau
    identity_n = np.eye(size_n, dtype=np.complex128)
    coherent_part = -1j * (np.kron(identity_n, hamiltonian) - np.kron(hamiltonian.T, identity_n))
    dissipative_part = np.diag(_dephasing_diagonal(size_n, gamma))
    return hamiltonian, coherent_part + dissipative_part


def build_liouvillian_sparse(tau: np.ndarray, coupling_j: complex, gamma: float) -> csc_matrix:
    """Build sparse Liouvillian in CSC format for large systems."""
    size_n = tau.shape[0]
    hamiltonian_sparse = -coupling_j / 2.0 * csc_matrix(tau, dtype=np.complex128)
    identity_n = identity(size_n, dtype=np.complex128, format="csc")
    coherent_part = -1j * (kron(identity_n, hamiltonian_sparse) - kron(hamiltonian_sparse.T, identity_n))
    dissipative_part = diags(_dephasing_diagonal(size_n, gamma), 0, format="csc")
    return (coherent_part + dissipative_part).asformat("csc")

