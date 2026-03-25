"""Physical observables and projections used by figure scripts."""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm
import warnings

from mpemba_refactored.liouvillian import density_diagonal_indices


def project_rho_on_modes(rho_initial: np.ndarray, left_vecs: np.ndarray, right_vecs: np.ndarray) -> np.ndarray:
    """Compute modal coefficients c_k using biorthogonal projection with `np.vdot`."""
    size_n = rho_initial.shape[0]
    rho_vec = rho_initial.flatten("F")
    mode_count = size_n * size_n
    coefficients = np.zeros(mode_count, dtype=np.complex128)

    for mode_index in range(mode_count):
        w_k = left_vecs[:, mode_index]
        v_k = right_vecs[:, mode_index]
        norm_factor = np.vdot(w_k, v_k)
        projection = np.vdot(w_k, rho_vec)
        coefficients[mode_index] = projection / norm_factor if not np.isclose(norm_factor, 0) else 0

    return coefficients


def mode_population_from_eigenvector(eigenvector: np.ndarray, size_n: int) -> np.ndarray:
    """Extract Re(diag(Mat(v_k))) with strict Fortran-order reshape."""
    rho_k = eigenvector.reshape((size_n, size_n), order="F")
    return np.diag(rho_k).real


def calculate_ipr_from_population(population: np.ndarray) -> float:
    """Compute IPR = Σ|P_i|^4 / (Σ|P_i|^2)^2."""
    abs_population = np.abs(population)
    sum_sq = np.sum(abs_population ** 2)
    if np.isclose(sum_sq, 0):
        return 0.0
    sum_quad = np.sum(abs_population ** 4)
    return float(sum_quad / (sum_sq ** 2))


def calculate_ipr_from_mode(eigenvector: np.ndarray, size_n: int) -> float:
    """Compute IPR directly from right eigenvector mode representation."""
    return calculate_ipr_from_population(mode_population_from_eigenvector(eigenvector, size_n))


def calculate_distance_metric(rho: np.ndarray) -> float:
    """Compute D(rho)=log(N)+Tr(rho log rho) used in Mpemba diagnostics."""
    size_n = rho.shape[0]
    if np.isclose(np.trace(np.dot(rho, rho)), 1.0):
        return float(np.log(size_n))

    with warnings.catch_warnings():
        # Legacy scripts suppress UserWarning emitted by scipy.linalg.logm for
        # nearly singular density matrices; we keep the same numerical behavior.
        warnings.simplefilter("ignore", UserWarning)
        log_rho = logm(rho)

    tr_rho_log_rho = np.trace(np.dot(rho, log_rho))
    return float(np.log(size_n) + tr_rho_log_rho.real)


def calculate_excitability_map_dense(left_vecs: np.ndarray, right_vecs: np.ndarray, mode_index: int, size_n: int) -> np.ndarray:
    """Compute B(k,i)=|c_k(rho_i)|^2 for local pure states rho_i=|i><i|."""
    w_k = left_vecs[:, mode_index]
    v_k = right_vecs[:, mode_index]
    norm_factor = np.vdot(w_k, v_k)
    if np.isclose(norm_factor, 0):
        return np.zeros(size_n)

    diag_indices = density_diagonal_indices(size_n)
    projections = w_k.conj()[diag_indices]
    return np.abs(projections / norm_factor) ** 2
