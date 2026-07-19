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



def analyze_liouvillian_modes_rightmost(liouvillian: csc_matrix, num_modes: int,
                                        t_transform: float = 25.0, resid_tol: float = 1e-8,
                                        ncv: int | None = None):
    """Rightmost (max Re lambda) eigenmodes via the exponential spectral transform.

    Motivation (referee fix CF-2): shift-invert at sigma~0 orders modes by |lambda| and
    therefore CANNOT find the slowest oscillating modes: on the regular 10x10 lattice the
    true rightmost quadruplet lambda = -0.0803 +/- 0.239i has 1173 modes closer to zero.
    Arnoldi on the propagator exp(L*T) instead orders modes by |exp(lambda*T)| = exp(Re(lambda)*T),
    i.e. exactly by Re(lambda), with no need to guess where the oscillating band lies.

    Implementation: matrix-free LinearOperator applying scipy's expm_multiply; eigenvalues are
    recovered from the Rayleigh quotient lambda = <v|L|v>/<v|v> (NOT from log of the transformed
    eigenvalue, which is Im-ambiguous mod 2*pi/T); every returned mode is validated by its
    relative residual ||L v - lambda v|| / ||v|| <= resid_tol. Returns (lambdas, vectors,
    residuals) sorted by decreasing Re(lambda); (None, None, None) on solver failure.

    t_transform sets the spectral separation e^{dRe*T}; T=25 (=2.5/gamma for gamma=0.1)
    suppresses the bulk (Re < -0.1) by >= e^{0.5} relative to the slow cluster while keeping
    the number of internal expm steps moderate. A generous Krylov subspace (default
    ncv = 4*num_modes) helps ARPACK split NEARLY degenerate multiplet copies; note that exact
    copies/conjugate partners may still be represented once — single-vector Krylov spaces
    contain one direction per distinct eigenvalue, which is sufficient for the sweep figures
    (they need the distinct eigenvalue groups plus a representative eigenvector per group;
    the lambda* partner has conjugate diagonal weights, so IPR/overlap are identical).
    """
    from scipy.sparse.linalg import expm_multiply

    n = liouvillian.shape[0]
    if ncv is None:
        ncv = min(n, max(4 * num_modes, 40))
    try:
        op = LinearOperator((n, n), dtype=np.complex128,
                            matvec=lambda x: expm_multiply(liouvillian * t_transform, x))
        mu, vecs = eigs(op, k=num_modes, which="LM", ncv=ncv)
    except Exception:
        return None, None, None

    lambdas = np.empty(num_modes, dtype=np.complex128)
    residuals = np.empty(num_modes, dtype=np.float64)
    for i in range(num_modes):
        v = vecs[:, i]
        v_norm2 = np.vdot(v, v).real
        lam = np.vdot(v, liouvillian @ v) / v_norm2
        lambdas[i] = lam
        residuals[i] = np.linalg.norm(liouvillian @ v - lam * v) / np.sqrt(v_norm2)
    if np.any(residuals > resid_tol):
        return None, None, None

    order = np.argsort(lambdas.real)[::-1]
    return lambdas[order], vecs[:, order], residuals[order]
