import numpy as np
from scipy.linalg import eig as scipy_eig
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import LinearOperator, eigs, splu


def analyze_liouvillian_modes_dense(liouvillian: np.ndarray):
    """Compute all right eigenmodes and sort by decreasing Re(lambda)."""
    eigenvalues, eigenvectors = np.linalg.eig(liouvillian)
    order = np.argsort(eigenvalues.real)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def analyze_liouvillian_modes_dense_strict(liouvillian: np.ndarray):
    """Compute biorthogonal basis (left/right eigenvectors) and sort by Re(lambda)."""
    eig_out = scipy_eig(liouvillian, left=True, right=True)
    eigenvalues = eig_out[0]
    left_eigenvectors = eig_out[1]
    right_eigenvectors = eig_out[2]
    order = np.argsort(eigenvalues.real)[::-1]
    return eigenvalues[order], left_eigenvectors[:, order], right_eigenvectors[:, order]


def analyze_liouvillian_modes_sparse_robust(liouvillian: csc_matrix, num_modes: int, sigma: complex = 1e-9 + 0j):
    """Shift-invert solver near lambda≈0.

    For A = L - sigma I, eigs is applied to A^{-1}; recovered eigenvalues are
    lambda = 1 / mu + sigma.
    """
    try:
        shifted = liouvillian - sigma * identity(liouvillian.shape[0], dtype=np.complex128, format="csc")
        lu = splu(shifted)
        inverse_operator = LinearOperator(liouvillian.shape, matvec=lu.solve, dtype=liouvillian.dtype)
        mu_values, mu_vectors = eigs(inverse_operator, k=num_modes, which="LM")
        lambda_values = 1.0 / mu_values + sigma
        order = np.argsort(lambda_values.real)[::-1]
        return lambda_values[order], mu_vectors[:, order]
    except Exception:
        return None, None


def get_biorthogonal_modes_sparse_strict(liouvillian: csc_matrix, num_modes: int, sigma: complex = 1e-9 + 0j):
    """Compute sparse right/left modes and match pairs by conjugate eigenvalues."""
    try:
        vals_r, vecs_r = eigs(liouvillian, k=num_modes, sigma=sigma, which="LM")
    except Exception:
        return None, None, None

    order = np.argsort(vals_r.real)[::-1]
    vals_r = vals_r[order]
    vecs_r = vecs_r[:, order]

    try:
        vals_l, vecs_l = eigs(liouvillian.getH(), k=num_modes, sigma=np.conj(sigma), which="LM")
    except Exception:
        return None, None, None

    final_left = np.zeros_like(vecs_r)
    for i, value_r in enumerate(vals_r):
        best = np.argmin(np.abs(vals_l - np.conj(value_r)))
        final_left[:, i] = vecs_l[:, best]

    return vals_r, vecs_r, final_left

