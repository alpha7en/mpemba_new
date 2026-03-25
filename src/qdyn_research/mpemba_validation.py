import os
import random

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .liouvillian import build_liouvillian_dense
from .metrics import calculate_excitability_map, entropic_distance, get_projection_coefficient
from .mpemba import find_guaranteed_mpemba_dense
from .spectral import analyze_liouvillian_modes_dense_strict
from .topology import generate_rewired_grid_tau_guaranteed_connectivity


# Spectral decomposition coefficient c_k = <w_k|rho>/<w_k|v_k> for non-Hermitian Liouvillian modes.
def worker_get_decomposition(rho_vec, evals, left_vecs, right_vecs):
    """Modal decomposition coefficients c_k = <w_k|rho> / <w_k|v_k>."""
    coeffs = []
    for k in range(len(evals)):
        coeffs.append(get_projection_coefficient(left_vecs[:, k], right_vecs[:, k], rho_vec))
    return np.array(coeffs)


def fast_diag_entropy(rho_vec, n):
    """Fast diagonal-state distance D = log(N) - S(p), S(p) = -sum_i p_i log p_i."""
    diag_indices = np.arange(n) * (n + 1)
    probs = rho_vec[diag_indices].real
    probs = probs[probs > 1e-15]
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs))
    return np.log(n) - entropy


def simulation_worker_spectral(rho_hot, rho_cold, time_points, n, evals, right_vecs, left_vecs):
    """Propagate two states via spectral expansion and detect crossing D_cold(t) - D_hot(t) > 0."""
    d_hot_0 = entropic_distance(rho_hot, n, reshape_order="C")
    d_cold_0 = entropic_distance(rho_cold, n, reshape_order="C")
    if d_hot_0 <= d_cold_0 + 1e-7:
        return False, None, 0.0

    c_hot = worker_get_decomposition(rho_hot, evals, left_vecs, right_vecs)
    c_cold = worker_get_decomposition(rho_cold, evals, left_vecs, right_vecs)

    crossed = False
    t_cross = None
    max_advantage = 0.0

    for t in time_points:
        decay = np.exp(evals * t)
        vec_h_t = right_vecs @ (c_hot * decay)
        vec_c_t = right_vecs @ (c_cold * decay)

        d_h = entropic_distance(vec_h_t, n, reshape_order="C")
        d_c = entropic_distance(vec_c_t, n, reshape_order="C")
        delta = d_c - d_h

        if delta > 0:
            if not crossed:
                crossed = True
                t_cross = t
            max_advantage = max(max_advantage, delta)

    return crossed, t_cross, max_advantage


class MpembaValidator:
    """Environment for testing Mpemba-state search on one rewired network realization."""

    def __init__(self, height=5, width=5, p=0.1, J=1.0, gamma=0.1):
        self.n = height * width
        self.params = {"h": height, "w": width, "p": p, "J": J, "gamma": gamma}

        self.tau = generate_rewired_grid_tau_guaranteed_connectivity(height, width, p)
        self.L = build_liouvillian_dense(self.tau, J, gamma)

        self.evals, self.left_vecs, self.right_vecs = analyze_liouvillian_modes_dense_strict(self.L)
        idx = np.argsort(self.evals.real)[::-1]
        self.evals = self.evals[idx]
        self.left_vecs = self.left_vecs[:, idx]
        self.right_vecs = self.right_vecs[:, idx]

        self.lambda_slow = None
        self.tau_sys = None
        self._compute_spectral_properties()

    def _compute_spectral_properties(self):
        """Estimate spectral gap from the two largest Re(lambda) values and define tau_sys = 1/|Re(lambda_1)|."""
        try:
            vals, _ = eigs(self.L, k=2, which="LR")
            idx = np.argsort(vals.real)[::-1]
            vals = vals[idx]
            if len(vals) > 1:
                self.lambda_slow = vals[1]
                real_part = abs(self.lambda_slow.real)
                self.tau_sys = 1.0 / real_part if real_part > 1e-12 else 1.0
            else:
                self.lambda_slow = -1.0 + 0j
                self.tau_sys = 1.0
        except ArpackNoConvergence:
            self.lambda_slow = -1.0 + 0j
            self.tau_sys = 1.0
        except Exception:
            self.lambda_slow = -1.0 + 0j
            self.tau_sys = 1.0

    def _get_decomposition(self, rho_vec):
        return worker_get_decomposition(rho_vec, self.evals, self.left_vecs, self.right_vecs)

    def generate_random_full_density_matrix(self, n=None):
        """Generate a random density matrix from the Hilbert-Schmidt measure (Ginibre ensemble)."""
        if n is None:
            n = self.n
        real_part = np.random.randn(n, n)
        imag_part = np.random.randn(n, n)
        g = real_part + 1j * imag_part
        rho = g @ g.conj().T
        rho /= np.trace(rho)
        return rho.flatten(order="C")

    def generate_random_diag_density_matrix(self):
        """Generate classical diagonal rho with random populations (sum_i p_i = 1)."""
        populations = np.random.random(self.n)
        populations /= np.sum(populations)
        rho = np.diag(populations).astype(complex)
        return rho.flatten(order="C")

    def compute_decay_time(self, rho_init_vec, decay_factor=np.e, t_max_multiplier=10, steps=50):
        """Find t such that D(t) = D(0)/decay_factor using ODE integration plus interpolation."""
        d0 = entropic_distance(rho_init_vec, self.n, reshape_order="C")
        if d0 < 1e-9:
            return 0.0
        d_target = d0 / decay_factor

        tau_sys = 1.0 / abs(self.lambda_slow.real) if self.lambda_slow is not None and abs(self.lambda_slow.real) > 1e-6 else 1.0
        t_max = tau_sys * t_max_multiplier
        t_eval = np.linspace(0, t_max, steps)

        y0 = np.concatenate([rho_init_vec.real, rho_init_vec.imag])
        n_sq = self.n ** 2

        def dydt(_t, y):
            vec = y[:n_sq] + 1j * y[n_sq:]
            d_vec = self.L.dot(vec)
            return np.concatenate([d_vec.real, d_vec.imag])

        sol = solve_ivp(dydt, (0, t_max), y0, t_eval=t_eval, method="RK45")

        d_vals = []
        for i in range(len(sol.t)):
            vec_t = sol.y[:n_sq, i] + 1j * sol.y[n_sq:, i]
            d_vals.append(entropic_distance(vec_t, self.n, reshape_order="C"))
        d_vals = np.array(d_vals)

        if d_vals[-1] > d_target:
            return None

        f_interp = interp1d(d_vals[::-1], sol.t[::-1], kind="linear")
        try:
            return float(f_interp(d_target))
        except ValueError:
            return None

    def calculate_overage_decay_time(self, iter=None):
        """Return mean/median/variance of decay times over a random diagonal-state ensemble."""
        if iter is None:
            iter = self.n ** 2 * 100
        times = []
        for _ in range(iter):
            rho = self.generate_random_diag_density_matrix()
            times.append(self.compute_decay_time(rho))
        times = np.array(times, dtype=np.float64)
        times = times[~np.isnan(times)]
        if times.size == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(times)), float(np.median(times)), float(np.var(times))

    def simulate_dynamics(self, rho_hot, rho_cold, time_points):
        """Evaluate D_hot(t), D_cold(t) on a time grid using rho(t)=V[ c * exp(lambda t) ]."""
        d_hot_0 = entropic_distance(rho_hot, self.n, reshape_order="C")
        d_cold_0 = entropic_distance(rho_cold, self.n, reshape_order="C")
        if d_hot_0 <= d_cold_0 + 1e-5:
            return False, None, 0.0

        c_hot = self._get_decomposition(rho_hot)
        c_cold = self._get_decomposition(rho_cold)

        crossed = False
        t_cross = None
        max_advantage = 0.0

        for t in time_points:
            decay = np.exp(self.evals * t)
            vec_h_t = self.right_vecs @ (c_hot * decay)
            vec_c_t = self.right_vecs @ (c_cold * decay)

            d_h = entropic_distance(vec_h_t, self.n, reshape_order="C")
            d_c = entropic_distance(vec_c_t, self.n, reshape_order="C")
            delta = d_c - d_h

            if delta > 0:
                if not crossed:
                    crossed = True
                    t_cross = t
                max_advantage = max(max_advantage, delta)

        return crossed, t_cross, max_advantage

    def run_smart_strategy(self, time_points):
        """Use ratio-optimized cold state and evaluate crossing statistics."""
        b_map = calculate_excitability_map(self.left_vecs, self.right_vecs, 1, self.n)
        vec_hot, vec_cold, _vec_cold_score = find_guaranteed_mpemba_dense(
            self.left_vecs,
            self.right_vecs,
            self.n,
            excitability_map=b_map,
            mode_idx=1,
            distance_order="C",
        )
        hot_dist = entropic_distance(vec_hot, self.n, reshape_order="C")
        cold_dist = entropic_distance(vec_cold, self.n, reshape_order="C")
        metric_gap = hot_dist - cold_dist
        success, t_cross, adv = self.simulate_dynamics(vec_hot, vec_cold, time_points)
        return success, t_cross, adv, metric_gap

    def run_smart_strategy_score(self, time_points):
        """Use score-optimized cold state and evaluate crossing statistics."""
        b_map = calculate_excitability_map(self.left_vecs, self.right_vecs, 1, self.n)
        vec_hot, _vec_cold, vec_cold_score = find_guaranteed_mpemba_dense(
            self.left_vecs,
            self.right_vecs,
            self.n,
            excitability_map=b_map,
            mode_idx=1,
            distance_order="C",
        )
        hot_dist = entropic_distance(vec_hot, self.n, reshape_order="C")
        cold_dist = entropic_distance(vec_cold_score, self.n, reshape_order="C")
        metric_gap = hot_dist - cold_dist
        success, t_cross, adv = self.simulate_dynamics(vec_hot, vec_cold_score, time_points)
        return success, t_cross, adv, metric_gap

    def run_smart_random_strategy(self, num_trials, time_points, mix_size_cold):
        """Random hot node versus random cold mixture with fixed mixture size."""
        successes = 0
        crossing_times = []

        for _ in range(num_trials):
            hot_node = random.randint(0, self.n - 1)
            rho_hot = np.zeros(self.n ** 2, dtype=complex)
            rho_hot[hot_node * self.n + hot_node] = 1.0

            candidates = list(range(self.n))
            candidates.remove(hot_node)
            cold_nodes = random.sample(candidates, mix_size_cold)

            rho_cold = np.zeros((self.n, self.n), dtype=complex)
            for node in cold_nodes:
                rho_cold[node, node] = 1.0 / mix_size_cold

            ok, t_cross, _adv = self.simulate_dynamics(rho_hot, rho_cold.flatten(order="C"), time_points)
            if ok:
                successes += 1
                crossing_times.append(t_cross)

        return successes, crossing_times

    def run_random_strategy(self, num_trials, time_points, metric_max=None, metric_min=0, metric_gap_min=0, metric_gap_max=None):
        """Brute-force random diagonal pairs under metric and gap constraints."""
        successes = 0
        crossing_times = []

        if metric_max is None:
            metric_max = np.log(self.n)
        if metric_gap_max is None:
            metric_gap_max = np.log(self.n)

        for _ in range(num_trials):
            rho_hot = self.generate_random_diag_density_matrix()
            rho_cold = self.generate_random_diag_density_matrix()

            hot_dist = entropic_distance(rho_hot, self.n, reshape_order="C")
            cold_dist = entropic_distance(rho_cold, self.n, reshape_order="C")
            if hot_dist <= cold_dist:
                rho_hot, rho_cold = rho_cold, rho_hot
                hot_dist, cold_dist = cold_dist, hot_dist

            metric_gap = hot_dist - cold_dist
            while not (metric_min <= cold_dist <= hot_dist <= metric_max and metric_gap_min <= metric_gap <= metric_gap_max):
                rho_hot = self.generate_random_diag_density_matrix()
                rho_cold = self.generate_random_diag_density_matrix()
                hot_dist = entropic_distance(rho_hot, self.n, reshape_order="C")
                cold_dist = entropic_distance(rho_cold, self.n, reshape_order="C")
                if hot_dist <= cold_dist:
                    rho_hot, rho_cold = rho_cold, rho_hot
                    hot_dist, cold_dist = cold_dist, hot_dist
                metric_gap = hot_dist - cold_dist

            ok, t_cross, _adv = self.simulate_dynamics(rho_hot, rho_cold, time_points)
            if ok:
                successes += 1
                crossing_times.append(t_cross)

        return successes, crossing_times

    def run_random_pull_strategy(self, num_trials, time_points, n_jobs=8, metric_max=None, metric_min=0, metric_gap_min=0, metric_gap_max=None):
        """Batch random search with vectorized gap filtering and parallel spectral simulation."""
        from joblib import Parallel, delayed

        successes = 0
        crossing_times = []

        if metric_max is None:
            metric_max = np.log(self.n)
        if metric_gap_max is None:
            metric_gap_max = np.log(self.n)

        batch_size = 2000

        while len(crossing_times) < num_trials:
            remaining = num_trials - len(crossing_times)
            pool_rhos = [self.generate_random_diag_density_matrix() for _ in range(batch_size)]
            pool_dists = np.array([fast_diag_entropy(rho, self.n) for rho in pool_rhos])

            valid = np.where((pool_dists >= metric_min) & (pool_dists <= metric_max))[0]
            if len(valid) < 2:
                continue

            filtered = pool_dists[valid]
            diff_matrix = filtered[:, np.newaxis] - filtered[np.newaxis, :]
            gap_mask = (diff_matrix >= metric_gap_min) & (diff_matrix <= metric_gap_max)
            hot_loc, cold_loc = np.where(gap_mask)
            if len(hot_loc) == 0:
                continue

            pair_indices = np.arange(len(hot_loc))
            np.random.shuffle(pair_indices)
            take_count = min(len(pair_indices), remaining * 3, num_trials // 2)

            batch_pairs = []
            for idx in pair_indices[:take_count]:
                hot_idx = valid[hot_loc[idx]]
                cold_idx = valid[cold_loc[idx]]
                batch_pairs.append((pool_rhos[hot_idx], pool_rhos[cold_idx]))

            results = Parallel(n_jobs=n_jobs)(
                delayed(simulation_worker_spectral)(
                    hot,
                    cold,
                    time_points,
                    self.n,
                    self.evals,
                    self.right_vecs,
                    self.left_vecs,
                )
                for hot, cold in batch_pairs
            )

            for ok, t_cross, _adv in results:
                if ok:
                    successes += 1
                    crossing_times.append(t_cross)
                if len(crossing_times) >= num_trials:
                    break

        return successes, crossing_times[:num_trials]

    def save_state(self, filename="validator_checkpoint.pkl"):
        import joblib

        joblib.dump(self, filename, compress=0)

    @staticmethod
    def load_state(filename="validator_checkpoint.pkl"):
        import joblib

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        return joblib.load(filename)

