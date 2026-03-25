from collections import deque
import time

import numpy as np
from scipy.integrate import solve_ivp


class QuantumSimulatorCore:
    @staticmethod
    def calculate_metric_d_from_vec(rho_vec: np.ndarray, n: int, log_n: float) -> float:
        """Evaluate D(rho)=log(N)-S(rho) from vec(rho) using column-major reshape."""
        rho_t = rho_vec.reshape((n, n), order="F")
        eigenvalues = np.linalg.eigvalsh(rho_t)
        entropy = -np.sum([val * np.log(val) for val in eigenvalues if val > 1e-15])
        return float(log_n - entropy)

    def run_simulation(
        self,
        liouvillian: np.ndarray,
        rho_0: np.ndarray,
        t_span=(0, 10),
        atol: float = 1e-4,
        patience: int = 10,
        num_plot_points: int = 500,
        method: str = "RK45",
    ):
        """Integrate d/dt vec(rho)=L vec(rho) and stop when D(t) reaches a plateau."""
        started = time.perf_counter()
        n = rho_0.shape[0]
        log_n = np.log(n)
        rho0_vec = rho_0.flatten(order="F").astype(np.complex128)

        def evolution(_t, rho_vec):
            return liouvillian @ rho_vec

        class PlateauTermination:
            def __init__(self, n_local, log_n_local, patience_local, atol_local):
                self.history = deque(maxlen=patience_local)
                self.patience = patience_local
                self.atol = atol_local
                self.n = n_local
                self.log_n = log_n_local

            def __call__(self, _t, rho_vec):
                current = QuantumSimulatorCore.calculate_metric_d_from_vec(rho_vec, self.n, self.log_n)
                self.history.append(current)
                if len(self.history) == self.patience and np.std(self.history) < self.atol:
                    return 0
                return 1

        plateau_event = PlateauTermination(n, log_n, patience, atol)
        plateau_event.terminal = True

        sol = solve_ivp(
            evolution,
            t_span,
            rho0_vec,
            method=method,
            events=plateau_event,
            dense_output=True,
        )

        t_end = sol.t[-1]
        t_plot = np.linspace(0, t_end, num_plot_points)
        rho_plot = sol.sol(t_plot)
        d_plot = [self.calculate_metric_d_from_vec(rho_plot[:, i], n, log_n) for i in range(rho_plot.shape[1])]

        elapsed = time.perf_counter() - started
        return t_plot, np.array(d_plot), elapsed

