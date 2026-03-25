import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigs, ArpackNoConvergence

from first_mode_analys import (
    find_guaranteed_mpemba_dense,
    entropic_distance,
    get_projection_coefficient,
    fast_diag_entropy
)
from reconecting.bright_dark.bright_dark_math import (
    build_liouvillian_dense,
    generate_rewired_tau_dense,
    analyze_modes_dense_strict
)


def worker_get_decomposition(rho_vec, evals, left_vecs, right_vecs):
    """Раскладывает состояние по собственным векторам: c_k."""
    coeffs = []
    # Проходим по всем модам (или по первым k, если хотим оптимизировать)
    for k in range(len(evals)):
        # Используем твою функцию проекции
        c_k = get_projection_coefficient(left_vecs[:, k], right_vecs[:, k], rho_vec)
        coeffs.append(c_k)
    return np.array(coeffs)


def simulation_worker_spectral(rho_hot, rho_cold, time_points, N, evals, right_vecs, left_vecs):
    """
    Воркер для параллельного запуска. Выполняет симуляцию одной пары состояний.

    Аргументы:
    rho_hot, rho_cold -- Векторизованные начальные состояния (векторы)
    time_points       -- Массив временных точек
    N                 -- Размерность решетки (сторона квадрата)
    evals             -- Вектор собственных чисел
    right_vecs        -- Матрица правых собственных векторов (R)
    left_vec --  Матрица левых собственных векторов (R)
    """

    # 1. Проверка начального условия (Hot должен быть дальше от равновесия)
    # Считаем энтропию. Важно: entropic_distance должна быть доступна в этой области видимости.
    d_hot_0 = entropic_distance(rho_hot, N)
    d_cold_0 = entropic_distance(rho_cold, N)

    # Если Hot уже "холоднее" (ближе к равновесию) или они равны — это не Мпемба.
    # Добавляем epsilon для стабильности.
    if d_hot_0 <= d_cold_0 + 1e-7:
        return False, None, 0.0

    # 2. Разложение по модам (Спектральная проекция)
    # c = <w | rho>. В матричном виде: C = Proj_Mat @ rho
    c_hot = worker_get_decomposition(rho_hot, evals, left_vecs, right_vecs)
    c_cold = worker_get_decomposition(rho_cold, evals, left_vecs, right_vecs)

    # 3. Эволюция во времени
    crossed = False
    t_cross = None
    max_advantage = 0.0

    # Чтобы не гонять лишние данные между процессами, массивы D_H и D_C не сохраняем,
    # если они не нужны для отладки. Ищем только факт пересечения.

    for t in time_points:
        # Вектор эволюции: e^(lambda * t)
        # evals - комплексные, t - вещественное
        decay = np.exp(evals * t)

        # Восстановление векторов плотности
        # rho(t) = sum(c_k * e^(lambda_k t) * v_k) -> R @ (c * decay)
        # Используем матричное умножение для скорости
        vec_h_t = right_vecs @ (c_hot * decay)
        vec_c_t = right_vecs @ (c_cold * decay)

        # Считаем метрики (расстояние до равновесия)
        d_h = entropic_distance(vec_h_t, N)
        d_c = entropic_distance(vec_c_t, N)

        # Вычисляем разницу (Cold - Hot)
        # Если delta > 0, значит d_c > d_h -> Hot стал ближе к 0 -> Пересечение!
        delta = d_c - d_h

        if delta > 0:
            if not crossed:
                crossed = True
                t_cross = t
            max_advantage = max(max_advantage, delta)
            # Опционально: можно прервать цикл (break), если нас интересует только
            # сам факт и время первого пересечения.
            # Если нужно найти max_advantage на всем интервале, break убираем.
            # break

    return crossed, t_cross, max_advantage


class MpembaValidator:
    """
    Класс для верификации алгоритмов поиска эффекта Мпембы.
    Инкапсулирует среду симуляции и методы сравнения.
    """

    def __init__(self, height=5, width=5, p=0.1, J=1.0, gamma=0.1):
        self.N = height * width
        self.params = {'h': height, 'w': width, 'p': p, 'J': J, 'gamma': gamma}
        self.max_entropy = np.log(self.N)

        print(f"--- Инициализация среды ({height}x{width}, p={p}) ---")
        # 1. Генерация системы (используем твои функции)
        self.tau = generate_rewired_tau_dense(height, width, p)
        self.L = build_liouvillian_dense(self.tau, J, gamma)

        # 2. Спектральный анализ (единоразово)
        # Предполагаем, что analyze_modes_dense_strict возвращает (evals, left, right)
        self.evals, self.left_vecs, self.right_vecs = analyze_modes_dense_strict(self.L)

        # Сортировка мод по Re(lambda) для удобства (0 - стационарная)
        idx = np.argsort(self.evals.real)[::-1]
        self.evals = self.evals[idx]
        self.left_vecs = self.left_vecs[:, idx]
        self.right_vecs = self.right_vecs[:, idx]

        self.lambda_slow = None  # Собственное число самой медленной моды
        self.tau_sys = None  # Характерное время релаксации

        # Сразу считаем спектральные свойства при создании объекта
        self._compute_spectral_properties()

        print(f"Система построена. Re(λ_1) = {self.evals[1].real:.4f}")

    def _get_decomposition(self, rho_vec):
        """Раскладывает состояние по собственным векторам: c_k."""
        coeffs = []
        # Проходим по всем модам (или по первым k, если хотим оптимизировать)
        for k in range(len(self.evals)):
            # Используем твою функцию проекции
            c_k = get_projection_coefficient(self.left_vecs[:, k], self.right_vecs[:, k], rho_vec)
            coeffs.append(c_k)
        return np.array(coeffs)

    def _compute_spectral_properties(self):
        """
        Внутренний метод. Находит самую медленную моду релаксации (Gap)
        и вычисляет характерное время системы tau_sys.
        """
        try:
            # Нам нужно найти собственные числа с наибольшей действительной частью.
            # В диссипативных системах Re(lambda) <= 0.
            # Самое большое - это 0 (стационар). Следующее за ним - lambda_slow.
            # Поэтому ищем k=2 моды с критерием 'LR' (Largest Real).

            # k=2: одна для стационарного состояния, одна для медленной моды
            vals, _ = eigs(self.L, k=2, which='LR')

            # Сортируем по убыванию действительной части:
            # Ожидаем: [~0.0, -0.012...]
            idx = np.argsort(vals.real)[::-1]
            sorted_vals = vals[idx]

            # Индекс 0 - это стационарное состояние (lambda approx 0)
            # Индекс 1 - это искомая медленная мода
            if len(sorted_vals) > 1:
                self.lambda_slow = sorted_vals[1]

                # Защита от деления на ноль, если спектр вырожден или ошибка точности
                real_part = abs(self.lambda_slow.real)
                if real_part > 1e-12:
                    self.tau_sys = 1.0 / real_part
                else:
                    print("Warning: Gap is too small, setting tau_sys = 1.0")
                    self.tau_sys = 1.0
            else:
                print("Warning: Could not find enough modes.")
                self.tau_sys = 1.0

        except ArpackNoConvergence as e:
            print(f"Spectral analysis failed: {e}")
            self.tau_sys = 1.0
            self.lambda_slow = -1.0 + 0j  # Заглушка
        except Exception as e:
            print(f"Error computing spectral properties: {e}")
            self.tau_sys = 1.0

    def generate_random_full_density_matrix(self, N=None):
        """
        Генерирует случайную матрицу плотности размера N x N.
        Распределение соответствует мере Гильберта-Шмидта.

        Аргументы:
        N -- размерность гильбертова пространства (int)

        Возвращает:
        rho -- валидная матрица плотности (numpy array complex128)
        """
        if not N:
            N = self.N
        # 1. Генерируем случайную комплексную матрицу G (Ансамбль Жинибра)
        # Элементы распределены нормально N(0, 1) + i*N(0, 1)
        real_part = np.random.randn(N, N)
        imag_part = np.random.randn(N, N)
        G = real_part + 1j * imag_part

        # 2. Делаем её положительно определенной и эрмитовой
        # M = G * G_dagger
        rho = G @ G.conj().T

        # 3. Нормируем след на 1
        rho /= np.trace(rho)

        rho_vec = rho.flatten()

        return rho_vec

    def generate_random_diag_density_matrix(self, N=None):
        """
        Генерирует случайную ДИАГОНАЛЬНУЮ матрицу плотности (без когерентностей).
        Физический смысл: классическое случайное распределение фотона по узлам.

        Возвращает: векторизованную матрицу (rho_vec).
        """
        if N is None:
            N = self.N

        # 1. Генерируем N случайных положительных чисел
        # np.random.random() дает числа от 0.0 до 1.0
        populations = np.random.random(N)

        # 2. Нормируем, чтобы сумма вероятностей была равна 1 (Trace = 1)
        populations /= np.sum(populations)

        # 3. Создаем диагональную матрицу
        # astype(complex) важен, так как решатели ожидают комплексные типы данных
        rho = np.diag(populations).astype(complex)

        # 4. Векторизуем
        return rho.flatten()

    def compute_decay_time(self, rho_init_vec, decay_factor=np.e, t_max_multiplier=10, steps=50):
        """
        Оптимизированный поиск времени затухания для разреженных матриц.

        Алгоритм:
        1. Симулируем динамику на грубой сетке (steps точек).
        2. Считаем энтропию только в этих точках (тяжелая операция).
        3. Ищем интерполяцией момент пересечения порога.
        """

        # 1. Целевая метрика
        # (Предполагаем, что entropic_distance уже импортирована)
        D0 = entropic_distance(rho_init_vec, self.N)
        if D0 < 1e-9: return 0.0
        D_target = D0 / decay_factor

        # 2. Оценка времени
        # Используем lambda_slow, если она есть, иначе берем наугад
        if hasattr(self, 'lambda_slow') and abs(self.lambda_slow.real) > 1e-6:
            tau_sys = 1.0 / abs(self.lambda_slow.real)
        else:
            tau_sys = 1.0

        t_max = tau_sys * t_max_multiplier
        t_eval = np.linspace(0, t_max, steps)

        # 3. Подготовка солвера (как в твоем коде)
        y0 = np.concatenate([rho_init_vec.real, rho_init_vec.imag])
        N_sq = self.N ** 2

        def dydt(t, y):
            # L - разреженная матрица, dot работает быстро
            vec = y[:N_sq] + 1j * y[N_sq:]
            d_vec = self.L.dot(vec)
            return np.concatenate([d_vec.real, d_vec.imag])

        # 4. Запуск интеграции (solve_ivp лучше контролирует ошибки, чем odeint)
        # method='RK45' - стандарт, 'BDF' - если система жесткая (большой разброс времен)
        sol = solve_ivp(dydt, (0, t_max), y0, t_eval=t_eval, method='RK45')

        # 5. Пост-процессинг (считаем энтропию только для steps точек)
        D_vals = []
        for i in range(len(sol.t)):
            vec_t = sol.y[:N_sq, i] + 1j * sol.y[N_sq:, i]
            # Важно: здесь мы восстанавливаем матрицу, это операция O(N^2),
            # а энтропия - O(N^3). Делаем это редко (50 раз), поэтому быстро.
            D_vals.append(entropic_distance(vec_t, self.N))

        D_vals = np.array(D_vals)

        # 6. Поиск времени пересечения (Интерполяция)
        # Нам нужно найти t, где D(t) == D_target
        # Поскольку D(t) монотонно убывает, это легко.

        if D_vals[-1] > D_target:
            print(f"Warning: Decay target not reached. Final D={D_vals[-1]:.4f}, Target={D_target:.4f}")
            return None  # Или sol.t[-1], если хотите вернуть "хотя бы что-то"

        # Создаем функцию интерполяции t(D) - обратная зависимость
        # D убывает, поэтому flip, чтобы x возрастал для интерполятора
        f_interp = interp1d(D_vals[::-1], sol.t[::-1], kind='linear')

        try:
            t_decay = float(f_interp(D_target))
            return t_decay
        except ValueError:
            return None

    def calculate_overage_decay_time(self, iter=None):
        """
        Считает усредненное время затухания для случайных матриц
        """
        if not iter:
            iter = self.N ** 2 * 100
        t = []
        for _ in range(iter):
            rho = self.generate_random_diag_density_matrix()
            t.append(self.compute_decay_time(rho))
        t = np.array(t)
        return np.mean(t), np.median(t), np.var(t)

    def simulate_dynamics(self, rho_hot, rho_cold, time_points):
        """
        Быстро симулирует эволюцию двух состояний и проверяет пересечение.
        Использует спектральное разложение: rho(t) = sum c_k * exp(lambda_k * t) * v_k
        """
        # 1. Проверка начального условия
        d_hot_0 = entropic_distance(rho_hot, self.N)
        d_cold_0 = entropic_distance(rho_cold, self.N)

        # if d_hot_0 <= d_cold_0 + 1e-5:
        #     d_hot_0, d_cold_0 = d_cold_0, d_hot_0

        if d_hot_0 <= d_cold_0 + 1e-5:
            return False, None, 0.0  # Hot должен быть дальше от равновесия

        # 2. Разложение по модам
        c_hot = self._get_decomposition(rho_hot)
        c_cold = self._get_decomposition(rho_cold)

        # 3. Эволюция во времени
        diffs = []
        crossed = False
        t_cross = None
        max_advantage = 0.0

        D_C = list()
        D_H = list()

        for t in time_points:
            # Вектор множителей exp(lambda * t)
            decay = np.exp(self.evals * t)

            # Восстановление векторов плотности
            # rho(t) = Right_Matrix @ (coeffs * decay)
            # Осторожно с размерностями numpy
            vec_h_t = self.right_vecs @ (c_hot * decay)
            vec_c_t = self.right_vecs @ (c_cold * decay)

            # Считаем расстояния (используем твою функцию)
            # Примечание: берем .real, так как расстояние вещественное, но численно может вылезти 0j
            d_h = entropic_distance(vec_h_t, self.N).real
            d_c = entropic_distance(vec_c_t, self.N).real
            D_H.append(d_h)
            D_C.append(d_c)
            delta = d_c - d_h  # Если > 0, значит Hot стал ближе (пересечение)
            diffs.append(delta)

            if delta > 0:
                if not crossed:
                    crossed = True
                    t_cross = t
                max_advantage = max(max_advantage, delta)
        # print(D_H)
        return crossed, t_cross, max_advantage

    def run_smart_strategy(self, time_points):
        """Запускает твой алгоритм поиска."""
        # Вызываем функцию из импортированного файла
        # mode_idx=1 соответствует самой медленной моде релаксации
        vec_hot, vec_cold, b_map, vec_cold_score = find_guaranteed_mpemba_dense(
            self.left_vecs, self.right_vecs, self.N, mode_idx=1
        )
        hot_dist = entropic_distance(vec_hot, self.N).real
        cold_dist = entropic_distance(vec_cold, self.N).real
        metric_gap = hot_dist - cold_dist
        print("smart gap", metric_gap)
        # print("Старое")
        # print(run_simulation_dense(self.L, vec_hot, t_max=30, steps=100)[1])
        # print("Новое")
        success, t_cross, adv = self.simulate_dynamics(vec_hot, vec_cold, time_points)
        # print("Пиво")
        return success, t_cross, adv, metric_gap

    def run_smart_strategy_score(self, time_points):
        """Запускает твой алгоритм поиска."""
        # Вызываем функцию из импортированного файла
        # mode_idx=1 соответствует самой медленной моде релаксации
        vec_hot, vec_cold, b_map, vec_cold_score = find_guaranteed_mpemba_dense(
            self.left_vecs, self.right_vecs, self.N, mode_idx=1
        )
        hot_dist = entropic_distance(vec_hot, self.N).real
        cold_dist_score = entropic_distance(vec_cold_score, self.N).real
        metric_gap = hot_dist - cold_dist_score
        print("smart gap", metric_gap)
        # print("Старое")
        # print(run_simulation_dense(self.L, vec_hot, t_max=30, steps=100)[1])
        # print("Новое")
        success, t_cross, adv = self.simulate_dynamics(vec_hot, vec_cold_score, time_points)
        # print("Пиво")
        return success, t_cross, adv, metric_gap

    def run_smart_random_strategy(self, num_trials, time_points, mix_size_cold):
        """
        Запускает случайный поиск.
        Hot: случайный чистый узел.
        Cold: случайная смесь mix_size_cold узлов.
        """
        successes = 0
        crossing_times = []

        for _ in range(num_trials):
            # Генерация случайного Hot
            hot_node = random.randint(0, self.N - 1)
            rho_hot = np.zeros(self.N ** 2, dtype=complex)
            rho_hot[hot_node * self.N + hot_node] = 1.0  # Векторизованный вид |i><i|

            # Генерация случайного Cold (смесь)
            # Исключаем hot_node, чтобы состояния были разными
            candidates = list(range(self.N))
            candidates.remove(hot_node)
            cold_nodes = random.sample(candidates, mix_size_cold)

            rho_cold_mat = np.zeros((self.N, self.N), dtype=complex)
            for node in cold_nodes:
                rho_cold_mat[node, node] = 1.0 / mix_size_cold
            rho_cold = rho_cold_mat.flatten()

            # Симуляция
            is_success, t_val, _ = self.simulate_dynamics(rho_hot, rho_cold, time_points)

            if is_success:
                successes += 1
                crossing_times.append(t_val)

        return successes, crossing_times

    def run_random_strategy(self, num_trials, time_points, metric_max=None, metric_min=0, metric_gap_min=0, metric_gap_max=None):
        """
        Запускает случайный поиск.
        Hot: случайная матрица.
        Cold: случайная матрица.
        """
        successes = 0
        crossing_times = []
        if not metric_max:
            metric_max = np.log(self.N)
        if not metric_gap_max:
            metric_gap_max = np.log(self.N)

        for _ in range(num_trials):
            print(_)
            rho_hot = self.generate_random_diag_density_matrix()
            rho_cold = self.generate_random_diag_density_matrix()
            hot_dist = entropic_distance(rho_hot, self.N).real
            cold_dist = entropic_distance(rho_cold, self.N).real
            if hot_dist <= cold_dist:
                rho_hot, rho_cold = rho_cold, rho_hot
                hot_dist, cold_dist = cold_dist, hot_dist
            metric_gap = hot_dist - cold_dist
            while not (metric_min <= cold_dist <= hot_dist <= metric_max and metric_gap_min <= metric_gap <= metric_gap_max):
                rho_hot = self.generate_random_diag_density_matrix()
                rho_cold = self.generate_random_diag_density_matrix()
                hot_dist = entropic_distance(rho_hot, self.N).real
                cold_dist = entropic_distance(rho_cold, self.N).real
                if hot_dist <= cold_dist:
                    rho_hot, rho_cold = rho_cold, rho_hot
                    hot_dist, cold_dist = cold_dist, hot_dist
                metric_gap = hot_dist - cold_dist
            # Симуляция
            is_success, t_val, _ = self.simulate_dynamics(rho_hot, rho_cold, time_points)

            if is_success:
                successes += 1
                crossing_times.append(t_val)

        return successes, crossing_times

    def run_random_pull_strategy(self, num_trials, time_points, n_jobs=-1, metric_max=None, metric_min=0, metric_gap_min=0,
                            metric_gap_max=None):
        """
        Оптимизированный случайный поиск (Batch strategy).
        Генерирует пул состояний, фильтрует их и находит пары с нужным зазором.
        """

        if not hasattr(self, 'projection_matrix'):
            # Создаем и кэшируем, если нет
            self.projection_matrix = self.left_vecs.conj().T

        successes = 0
        crossing_times = []

        # Установка дефолтных значений
        if metric_max is None: metric_max = np.log(self.N)
        if metric_gap_max is None: metric_gap_max = np.log(self.N)

        BATCH_SIZE = 2000

        print(f"Starting batch generation (target trials: {num_trials})...")

        while len(crossing_times) < num_trials:
            remaining = num_trials - len(crossing_times)
            print("Итерация", remaining)
            # 1. Генерируем пул матриц и считаем их метрики
            # Можно распараллелить или просто списком
            pool_rhos = [self.generate_random_diag_density_matrix() for _ in range(BATCH_SIZE)]

            # Считаем дистанции.
            pool_dists = np.array([fast_diag_entropy(rho, self.N) for rho in pool_rhos])

            # 2. Фильтрация по абсолютным границам (metric_min/max)
            # Оставляем только те индексы, которые попадают в диапазон
            mask_valid_range = (pool_dists >= metric_min) & (pool_dists <= metric_max)

            valid_indices = np.where(mask_valid_range)[0]
            if len(valid_indices) < 2:
                continue

            filtered_dists = pool_dists[valid_indices]

            # 3. Поиск пар с нужным Gap (Векторизованный подход)
            # Создаем матрицу разностей: Diff[i, j] = Dist[i] - Dist[j]
            # Используем broadcasting: (N, 1) - (1, N)
            diff_matrix = filtered_dists[:, np.newaxis] - filtered_dists[np.newaxis, :]

            # Условия:
            # 1. Gap в нужном диапазоне
            # 2. i != j (не та же самая матрица)
            gap_mask = (diff_matrix >= metric_gap_min) & \
                       (diff_matrix <= metric_gap_max)

            # Получаем индексы (i, j) в локальном массиве filtered_dists
            hot_loc_idxs, cold_loc_idxs = np.where(gap_mask)

            if len(hot_loc_idxs) == 0:
                continue

            batch_pairs = []

            # 4. Выбор пар
            # Перемешиваем найденные пары, чтобы избежать корреляций
            pair_indices = np.arange(len(hot_loc_idxs))
            np.random.shuffle(pair_indices)

            # Добавляем в общий список
            take_count = min(len(pair_indices), remaining * 3, num_trials // 2)
            print(take_count)
            for idx in pair_indices[:take_count]:
                real_hot_idx = valid_indices[hot_loc_idxs[idx]]
                real_cold_idx = valid_indices[cold_loc_idxs[idx]]
                batch_pairs.append((pool_rhos[real_hot_idx], pool_rhos[real_cold_idx]))

            if not batch_pairs:
                continue
            print("полетели")
            results = Parallel(n_jobs=8)(
                delayed(simulation_worker_spectral)(
                    hot,
                    cold,
                    time_points,
                    self.N,
                    self.evals,
                    self.right_vecs,
                    self.left_vecs
                ) for hot, cold in batch_pairs
            )
            for is_success, t_val, error in results:
                if is_success:
                    successes += 1
                    crossing_times.append(t_val)

            if len(crossing_times) >= num_trials:
                # Обрезаем лишнее, если перебрали
                crossing_times = crossing_times[:num_trials]
                break
        return successes, crossing_times

    def save_state(self, filename="validator_checkpoint.pkl"):
        """
        Сохраняет текущий экземпляр класса (со всей матрицей L, спектром и N) в файл.
        """
        print(f"Saving validator state to {filename}...")
        joblib.dump(self, filename, compress=0)  # compress=3 дает хороший баланс скорости и сжатия
        print("Saved.")

    @staticmethod
    def load_state(filename="validator_checkpoint.pkl"):
        """
        Загружает экземпляр класса из файла.
        Использование: validator = MpembaValidator.load_state("filename.pkl")
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден.")

        print(f"Loading validator state from {filename}...")
        obj = joblib.load(filename)
        print("Loaded successfully.")
        return obj
# --- БЛОК ЗАПУСКА ТЕСТА ---


if __name__ == "__main__":
    with open('res_gap/pivoprosto.txt', 'w', encoding='utf-8') as f:
        # Параметры теста
        H, W = 6, 6
        P_REWIRE = 0.15  # Достаточно большое p, чтобы эффекты локализации проявились
        NUM_RANDOM_TRIALS = 5000
        TIME_HORIZON = np.linspace(0, 30, 300)  # Время наблюдения

        # Создаем валидатор (он сам подтянет твои функции для генерации и анализа)
        validator = MpembaValidator(height=H, width=W, p=P_REWIRE)
        print("max metric gap", np.log(validator.N), file=f)
        print(validator.tau_sys, file=f)
        validator.save_state("res_gap/pivoprosto3.pkl")
        # print(validator.calculate_overage_decay_time(100))
        print(f"\n=== ЗАПУСК БЕНЧМАРКА (Test 1) ===", file=f)
        # 1. Тестируем НАШ алгоритм
        print("\n>>> Проверка Smart Algorithm...", file=f)
        smart_ok, smart_time, smart_adv, smart_gap = validator.run_smart_strategy_score(TIME_HORIZON)
        # smart_ok_score, smart_time_score, smart_adv_score, smart_gap_score = validator.run_smart_strategy_score(TIME_HORIZON)
        # print("ratio", smart_ok, smart_time, smart_adv, smart_gap)
        # print("score", smart_ok_score, smart_time_score, smart_adv_score, smart_gap_score)
        if smart_ok:
            print(f"[SUCCESS] Алгоритм нашел эффект!", file=f)
            print(f"  Время пересечения t*: {smart_time:.2f}", file=f)
            print(f"  Сила эффекта (advantage): {smart_adv:.4f}", file=f)
        else:
            print(f"[FAIL] Алгоритм не смог найти эффект (или условия не выполнены).", file=f)

        # 2. Тестируем RANDOM
        # Берем размер смеси для Cold таким же, как обычно находит наш алгоритм (например, 3-5)
        # MIX_SIZE = 4
        print(f"\n>>> Проверка Random Search ({NUM_RANDOM_TRIALS} попыток)...", file=f)
        divide = 10

        rnd_ok_count, rnd_times = validator.run_random_pull_strategy(
            NUM_RANDOM_TRIALS, TIME_HORIZON, metric_gap_min=min(np.log(validator.N), smart_gap) / divide
        )

        """
        for gap_num in range(divide):
            metric_gap_min = gap_num * min(np.log(validator.N), 2 * smart_gap)/divide
            metric_gap_max = (gap_num + 1) * min(np.log(validator.N), 2 * smart_gap) / divide
            print(metric_gap_min, metric_gap_max)
            rnd_ok_count, rnd_times = validator.run_random_pull_strategy(
                NUM_RANDOM_TRIALS, TIME_HORIZON, metric_gap_min=metric_gap_min, metric_gap_max=metric_gap_max
            )
    
            rnd_rate = (rnd_ok_count / NUM_RANDOM_TRIALS) * 100
            print(f"  Найдено успехов: {rnd_ok_count}/{NUM_RANDOM_TRIALS} ({rnd_rate:.1f}%)")
    
            # 3. Визуализация и выводы
            print("\n=== РЕЗУЛЬТАТЫ ===")
            efficiency = 100.0 / (rnd_rate if rnd_rate > 0 else 0.01)
    
            if smart_ok:
                print(f"Наш метод эффективнее случайного в {efficiency:.1f} раз.")
    
            plt.figure(figsize=(10, 6))
            plt.hist(rnd_times, bins=20, color='gray', alpha=0.6, label='Случайные находки')
    
            if smart_ok:
                plt.axvline(smart_time, color='red', linestyle='--', linewidth=3, label='Наш алгоритм')
                plt.text(smart_time, 1, f' t*={smart_time:.2f}', color='red', fontweight='bold')
    
            plt.title(f"Сравнение поиска эффекта Мпембы\nСетка {H}x{W}, p={P_REWIRE}")
            plt.xlabel("Время пересечения t*")
            plt.ylabel("Частота")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"res_gap/{gap_num}.png")
        """

        rnd_rate = (rnd_ok_count / NUM_RANDOM_TRIALS) * 100
        print(f"  Найдено успехов: {rnd_ok_count}/{NUM_RANDOM_TRIALS} ({rnd_rate:.1f}%)", file=f)

        # 3. Визуализация и выводы
        print("\n=== РЕЗУЛЬТАТЫ ===", file=f)
        efficiency = 100.0 / (rnd_rate if rnd_rate > 0 else 0.01)

        if smart_ok:
            print(f"Наш метод эффективнее случайного в {efficiency:.1f} раз.", file=f)

        plt.figure(figsize=(10, 6))
        plt.hist(rnd_times, bins=100, color='gray', alpha=0.6, label='Случайные находки')
        print(rnd_times, file=f)

        if smart_ok:
            plt.axvline(smart_time, color='red', linestyle='--', linewidth=3, label='Наш алгоритм')
            # plt.axvline(smart_time_score, color='green', linestyle='--', linewidth=3, label='Наш алгоритм с другим score')
            plt.text(smart_time, 1, f' t*={smart_time:.2f}', color='red', fontweight='bold')

        plt.title(f"Сравнение поиска эффекта Мпембы\nСетка {H}x{W}, p={P_REWIRE}")
        plt.xlabel("Время пересечения t*")
        plt.ylabel("Частота")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"res_gap/pivoprosto.png")
