import warnings

import numpy as np
from scipy.linalg import logm, lu_factor, lu_solve


def get_projection_coefficient(left_vec: np.ndarray, right_vec: np.ndarray, state_vec: np.ndarray):
    """Projection coefficient c_k = <w_k|rho> / <w_k|v_k> with Hermitian inner product (vdot)."""
    numerator = np.vdot(left_vec, state_vec)
    denominator = np.vdot(left_vec, right_vec)
    if np.abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def project_rho_on_modes(rho_initial: np.ndarray, left_vecs: np.ndarray, right_vecs: np.ndarray, order: str = "F"):
    """Compute all modal coefficients for vec(rho), typically with column-major vectorization (order='F')."""
    n = rho_initial.shape[0]
    rho_vec = rho_initial.flatten(order)
    coeffs = np.zeros(n * n, dtype=np.complex128)
    for k in range(n * n):
        w_k = left_vecs[:, k]
        v_k = right_vecs[:, k]
        norm = np.vdot(w_k, v_k)
        proj = np.vdot(w_k, rho_vec)
        coeffs[k] = proj / norm if not np.isclose(norm, 0) else 0.0
    return coeffs


def factor_mode_basis(right_vecs: np.ndarray):
    """LU factorization of the right-eigenvector matrix V, for exact modal projections."""
    return lu_factor(right_vecs)


def project_rho_on_modes_exact(rho_initial: np.ndarray, lu_piv, right_vecs: np.ndarray, order: str = "F"):
    """Exact modal coefficients: solve V c = vec(rho) instead of pairwise biorthogonal division.

    Pairwise division c_k = <w_k|rho>/<w_k|v_k> is invalid inside degenerate multiplets
    (LAPACK left/right eigenvectors are not biorthogonally paired there; on regular lattices
    the reconstruction error can exceed 1e3 %). Solving the linear system V c = vec(rho) is the
    exact decomposition regardless of degeneracy and needs no left eigenvectors.

    Returns (coeffs, residual) where residual = ||V c - vec(rho)|| / ||vec(rho)|| is the
    reconstruction-error control (should be ~1e-12; large values signal a defective basis).

    Up to two steps of iterative refinement (re-using the LU) are applied: for states with a
    large component in a nearly defective multiplet the raw backward error is eps*||V||*||c||
    with ||c|| up to ~1e8, and refinement recovers most of the lost digits.
    """
    rho_vec = rho_initial.flatten(order)
    b_norm = np.linalg.norm(rho_vec)
    coeffs = lu_solve(lu_piv, rho_vec)
    residual_vec = right_vecs @ coeffs - rho_vec
    residual = float(np.linalg.norm(residual_vec) / b_norm)
    for _ in range(2):
        if residual < 1e-12:
            break
        coeffs = coeffs - lu_solve(lu_piv, residual_vec)
        residual_vec = right_vecs @ coeffs - rho_vec
        new_residual = float(np.linalg.norm(residual_vec) / b_norm)
        if new_residual >= residual:      # refinement stalled at the attainable floor
            break
        residual = new_residual
    return coeffs, residual


def calculate_excitability_map(left_vecs: np.ndarray, right_vecs: np.ndarray, k_idx: int, n: int):
    """Local excitability map B(k,i)=|<w_k|rho_i>/<w_k|v_k>|^2 for rho_i=|i><i|."""
    w_k = left_vecs[:, k_idx]
    v_k = right_vecs[:, k_idx]
    norm = np.vdot(w_k, v_k)
    if np.isclose(norm, 0):
        return np.zeros(n)
    diag_indices = np.arange(n) * (n + 1)
    projections = w_k[diag_indices].conj()
    return np.abs(projections / norm) ** 2


def calculate_ipr(eigenvector: np.ndarray, n: int, reshape_order: str = "F") -> float:
    """IPR = sum(|p_i|^4) / (sum(|p_i|^2))^2 for population diagonal p_i of mode matrix."""
    rho_k = eigenvector.reshape((n, n), order=reshape_order)
    populations = np.abs(np.diag(rho_k))
    sum_sq = np.sum(populations ** 2)
    if np.isclose(sum_sq, 0):
        return 0.0
    sum_quad = np.sum(populations ** 4)
    return float(sum_quad / (sum_sq ** 2))


def entropic_distance(rho_vec: np.ndarray, n: int, reshape_order: str = "C") -> float:
    """Entropy distance D = log(N) - S(rho), with S(rho) = -Tr(rho log rho)."""
    rho = rho_vec.reshape((n, n), order=reshape_order)
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    evals = evals / np.sum(evals)
    entropy = -np.sum(evals * np.log(evals))
    return float(np.log(n) - entropy)


def calculate_distance_metric_logm(rho: np.ndarray) -> float:
    """Distance metric D = log(N) + Tr(rho log rho) computed through matrix logarithm."""
    n = rho.shape[0]
    if np.isclose(np.trace(np.dot(rho, rho)), 1.0):
        return float(np.log(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        log_rho = logm(rho)
    tr_term = np.trace(np.dot(rho, log_rho))
    return float(np.log(n) + tr_term.real)

