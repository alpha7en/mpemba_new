import numpy as np
import scipy.linalg as sl
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import deque
import time

# --- БЛОК I: ОПТИМИЗИРОВАННОЕ ЯДРО ---

class QuantumSimulatorCore:
    """
    Оптимизированное ядро для моделирования квантовой динамики
    в сетевых системах.
    """

    @staticmethod
    def build_liouvillian(tau, J, gamma):
        """
        Собирает матрицу Лиувилляна L из физических параметров.

        Args:
            tau (np.ndarray): Матрица смежности сети (N x N).
            J (float): Сила связи (туннелирование).
            gamma (float): Скорость дефазировки (диссипация).

        Returns:
            np.ndarray: Комплексная матрица Лиувилляна (N^2 x N^2).
        """
        N = tau.shape[0]
        dim_L = N**2

        # 1. Гамильтониан H
        H = -J / 2 * tau

        # 2. Гамильтонова часть L_H
        I_N = np.identity(N)
        L_H = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))

        # 3. Диссипативная часть L_D
        L_D_diag = np.zeros(dim_L, dtype=np.complex128)
        # Векторизованное создание диагонали L_D
        off_diag_indices = np.where(np.eye(N) == 0)
        k_indices = off_diag_indices[0] + off_diag_indices[1] * N
        L_D_diag[k_indices] = -gamma
        L_D = np.diag(L_D_diag)

        return L_H + L_D

    @staticmethod
    def _calculate_metric_D_from_vec(rho_vec, N, log_N):
        """Вспомогательная функция для расчета метрики из вектора."""
        rho_t = rho_vec.reshape((N, N), order='F')
        eigenvalues = np.linalg.eigvalsh(rho_t)
        entropy = -np.sum([val * np.log(val) for val in eigenvalues if val > 1e-15])
        return log_N - entropy

    def run_simulation(self, L, rho_0, t_span=(0, 10), atol=1e-4, patience=10, num_plot_points=500, method='RK45'):
        """
        Запускает моделирование с адаптивным шагом и динамическим завершением.
        Возвращает гладкую кривую эволюции с заданным числом точек.

        Args:
            L (np.ndarray): Матрица Лиувилляна.
            rho_0 (np.ndarray): Начальная матрица плотности.
            t_span (tuple): Максимальный интервал времени.
            atol (float): Толерантность для определения плато.
            patience (int): Количество шагов для проверки выхода на плато.
            num_plot_points (int): Количество точек для построения гладкого графика.

        Returns:
            tuple: (массив времен, массив метрики D(t), время выполнения).
        """
        t_start = time.perf_counter()

        N = rho_0.shape[0]
        log_N = np.log(N)
        rho0_vec = rho_0.flatten(order='F').astype(np.complex128)

        def evolution(t, rho_vec):
            return L @ rho_vec

        class PlateauTermination:
            def __init__(self, N, log_N, patience, atol):
                self.history = deque(maxlen=patience)
                self.patience = patience
                self.atol = atol
                self.N = N
                self.log_N = log_N

            def __call__(self, t, rho_vec):
                current_D = QuantumSimulatorCore._calculate_metric_D_from_vec(rho_vec, self.N, self.log_N)
                self.history.append(current_D)
                if len(self.history) == self.patience:
                    std_dev = np.std(self.history)
                    if std_dev < self.atol:
                        return 0
                return 1

        plateau_event = PlateauTermination(N, log_N, patience, atol)
        plateau_event.terminal = True

        # --- ЭТАП 1: Оптимальное решение ОДУ ---
        # Решатель находит непрерывное решение и останавливается по событию.
        sol = solve_ivp(
            evolution,
            t_span,
            rho0_vec,
            method=method,
            events=plateau_event,
            dense_output=True # Это ключ к получению гладкой кривой
        )

        # --- ЭТАП 2: Запрос данных для гладкого графика ---
        # Определяем фактическое время окончания симуляции
        t_end_actual = sol.t[-1]

        # Создаем плотную временную сетку от 0 до фактического конца
        t_plot = np.linspace(0, t_end_actual, num_plot_points)

        # Используем "плотный вывод" (sol.sol) для получения значений
        # на нашей плотной сетке. Это высокоточная интерполяция.
        rho_vecs_plot = sol.sol(t_plot)

        # Расчет метрики D для всех точек на нашей гладкой сетке
        D_plot = [self._calculate_metric_D_from_vec(rho_vecs_plot[:, i], N, log_N)
                  for i in range(rho_vecs_plot.shape[1])]

        t_end_perf = time.perf_counter()
        elapsed_time = t_end_perf - t_start

        # Возвращаем плотную сетку и соответствующие ей значения метрики
        return t_plot, np.array(D_plot), elapsed_time


# Создаем экземпляр нашего ядра для использования в других блоках
if __name__=="main":
    core = QuantumSimulatorCore()