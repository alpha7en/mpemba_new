import unittest
import random

import numpy as np

from mpemba_refactored.liouvillian import build_liouvillian_dense
from mpemba_refactored.npz_format import pack_run_result
from mpemba_refactored.observables import project_rho_on_modes
from mpemba_refactored.topology import (
    check_connectivity,
    generate_grid_tau,
    generate_rewired_grid_tau_guaranteed_connectivity,
)


class RefactoredCoreTests(unittest.TestCase):
    def test_generate_grid_tau_node_count_and_symmetry(self):
        tau = generate_grid_tau(3, 4)
        self.assertEqual(tau.shape, (12, 12))
        self.assertTrue(np.array_equal(tau, tau.T))

    def test_rewired_topology_remains_connected(self):
        random.seed(123)
        np.random.seed(123)
        tau = generate_rewired_grid_tau_guaranteed_connectivity(4, 4, p=1.0)
        self.assertTrue(check_connectivity(tau))

    def test_liouvillian_dephasing_preserves_population_indices(self):
        tau = generate_grid_tau(2, 2)
        _, liouvillian = build_liouvillian_dense(tau, coupling_j=1.0, gamma=0.1)
        diagonal = np.diag(liouvillian)
        population_indices = np.arange(4) * (4 + 1)
        for idx in population_indices:
            self.assertAlmostEqual(diagonal[idx].real, 0.0, places=12)
            self.assertAlmostEqual(diagonal[idx].imag, 0.0, places=12)

    def test_projection_uses_biorthogonal_vdot_convention(self):
        rho_initial = np.array([[1.0 + 0.0j, 0.0], [0.0, 0.0]], dtype=np.complex128)
        left_vecs = np.eye(4, dtype=np.complex128)
        right_vecs = np.eye(4, dtype=np.complex128)
        left_vecs[:, 0] = np.array([1j, 0, 0, 0], dtype=np.complex128)
        right_vecs[:, 0] = np.array([1.0, 0, 0, 0], dtype=np.complex128)

        coefficients = project_rho_on_modes(rho_initial, left_vecs, right_vecs)
        # With np.vdot in both numerator and denominator:
        # <w|rho> = -1j, <w|v> = -1j => c0 = 1
        self.assertAlmostEqual(coefficients[0].real, 1.0, places=12)
        self.assertAlmostEqual(coefficients[0].imag, 0.0, places=12)

    def test_pack_run_result_keeps_legacy_npz_key_schema(self):
        lambdas = np.array([0.0 + 0j, -0.1 + 0j])
        vectors = np.zeros((9, 2), dtype=np.complex128)
        packed = pack_run_result(0.01, 7, lambdas, vectors)
        self.assertIn("p_0.01_run_7_lambdas", packed)
        self.assertIn("p_0.01_run_7_vectors", packed)
        self.assertIn("p_0.01_run_7_p_value", packed)


if __name__ == "__main__":
    unittest.main()

