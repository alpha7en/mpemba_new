import numpy as np
from scipy.sparse import csc_matrix, diags, identity, kron


def build_liouvillian_dense(tau: np.ndarray, J: complex, gamma: float) -> np.ndarray:
    """Build dense Liouvillian

    L = -i (I ⊗ H - H^T ⊗ I) + L_D,  H = -(J/2) tau.
    L_D applies pure dephasing: diagonal density-matrix elements have 0 decay,
    off-diagonal elements decay with rate gamma.
    """
    tau_dense = np.asarray(tau, dtype=np.complex128)
    n = tau_dense.shape[0]
    hamiltonian = -J / 2.0 * tau_dense
    identity_n = np.eye(n, dtype=np.complex128)
    coherent = -1j * (np.kron(identity_n, hamiltonian) - np.kron(hamiltonian.T, identity_n))

    diagonal = np.full(n * n, -gamma, dtype=np.complex128)
    idx = np.arange(n)
    diagonal[idx * n + idx] = 0.0
    dissipative = np.diag(diagonal)
    return coherent + dissipative


def build_liouvillian_sparse(tau: np.ndarray, J: complex, gamma: float) -> csc_matrix:
    """Sparse version of the same Liouvillian formula used for large N systems."""
    n = tau.shape[0]
    tau_sparse = csc_matrix(tau, dtype=np.complex128)
    hamiltonian = -J / 2.0 * tau_sparse
    identity_n = identity(n, dtype=np.complex128, format="csc")
    coherent = -1j * (kron(identity_n, hamiltonian) - kron(hamiltonian.T, identity_n))

    diagonal = np.full(n * n, -gamma, dtype=np.complex128)
    idx = np.arange(n)
    diagonal[idx * n + idx] = 0.0
    dissipative = diags(diagonal, 0, format="csc")
    return (coherent + dissipative).asformat("csc")

