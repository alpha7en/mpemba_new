import numpy as np
import os
import datetime
import random
from collections import deque
from scipy.sparse import csc_matrix, identity, kron, diags
from scipy.sparse.linalg import eigs, splu, LinearOperator
import concurrent.futures
import time


# --- ============================================= ---
# --- ЯДРО: Проверенные и надежные функции          ---
# --- ============================================= ---

def check_connectivity(tau: np.ndarray) -> bool:
    """Проверяет связность графа. Возвращает True, если граф связный."""
    N = tau.shape[0]
    if N == 0: return True
    visited = set()
    q = deque([0])  # Начинаем обход с узла 0
    visited.add(0)
    count = 0
    while q:
        u = q.popleft()
        count += 1
        neighbors = np.where(tau[u, :] == 1)[0]
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                q.append(v)
    # Если число посещенных узлов равно общему числу, граф связный
    return count == N


def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Генерация базовой решетки."""
    N = height * width;
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        if (i + 1) % width != 0: tau[i, i + 1] = tau[i + 1, i] = 1
        if i < N - width: tau[i, i + width] = tau[i + width, i] = 1
    return tau


def generate_rewired_grid_tau_GUARANTEED_CONNECTIVITY(height: int, width: int, p: float) -> np.ndarray:
    """
    Генерация сети по алгоритму перестроения с гарантией итоговой связности.
    """
    while True:  # Цикл для перегенерации в случае разрыва графа
        initial_tau = generate_grid_tau(height, width)
        tau = initial_tau.copy()
        N = height * width
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if initial_tau[i, j] == 1]
        random.shuffle(edges)

        for u, v in edges:
            if random.random() < p:
                node_to_rewire = u if random.random() < 0.5 else v
                original_partner = v if node_to_rewire == u else u

                neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
                forbidden_nodes = set(neighbors) | {node_to_rewire, original_partner}
                valid_targets = [n for n in range(N) if n not in forbidden_nodes]

                if valid_targets:
                    w = random.choice(valid_targets)
                    tau[u, v] = tau[v, u] = 0
                    tau[node_to_rewire, w] = tau[w, node_to_rewire] = 1

        # Проверяем связность. Если все хорошо, выходим из цикла.
        if check_connectivity(tau):
            return tau
        # Если граф разорван, цикл начнется заново.


def build_liouvillian_sparse(tau: np.ndarray, J: complex, gamma: float) -> csc_matrix:
    """Построение разреженного Лиувиллиана."""
    N = tau.shape[0];
    H = -J / 2.0 * csc_matrix(tau, dtype=np.complex128)
    I = identity(N, dtype=np.complex128, format='csc');
    L_H = -1j * (kron(I, H) - kron(H.T, I))
    diag_LD = np.full(N * N, -gamma, dtype=np.complex128);
    diag_LD[np.arange(N) * N + np.arange(N)] = 0.0
    return (L_H + diags(diag_LD, 0, format='csc')).asformat('csc')


def analyze_liouvillian_modes_sparse_ROBUST(L: csc_matrix, num_modes: int):
    """
    Находит `num_modes` самых медленных мод, используя надежный метод
    сдвига и инверсии (Shift-and-Invert). Возвращает и значения, и векторы.
    """
    sigma = 1e-9 + 0j
    try:
        lu = splu(L - sigma * identity(L.shape[0], dtype=np.complex128, format='csc'))
        op_inv = LinearOperator(L.shape, matvec=lu.solve, dtype=L.dtype)

        # Запрашиваем и собственные значения (mu), и собственные векторы
        mu_values, mu_vectors = eigs(op_inv, k=num_modes, which='LM')

        lambda_values = 1.0 / mu_values + sigma

        # Сортируем и значения, и векторы
        sort_indices = np.argsort(lambda_values.real)[::-1]

        return lambda_values[sort_indices], mu_vectors[:, sort_indices]

    except Exception as e:
        return None, None


def single_run(p, run_idx, HEIGHT, WIDTH, J, gamma, NUM_MODES_TO_FIND):
    tau = generate_rewired_grid_tau_GUARANTEED_CONNECTIVITY(HEIGHT, WIDTH, p)
    L_sparse = build_liouvillian_sparse(tau, J, gamma)
    lambdas, vectors = analyze_liouvillian_modes_sparse_ROBUST(L_sparse, num_modes=NUM_MODES_TO_FIND)

    if lambdas is not None:
        key_prefix = f"p_{p}_run_{run_idx}"
        return {
            f"{key_prefix}_lambdas": lambdas,
            f"{key_prefix}_vectors": vectors,
            f"{key_prefix}_p_value": p
        }
    else:
        return {}


# --- ============================================= ---
# --- ОСНОВНОЙ БЛОК: Пакетная симуляция и сохранение ---
# --- ============================================= ---
if __name__ == "__main__":
    # 1. ОБЩИЕ ПАРАМЕТРЫ СИМУЛЯЦИИ
    HEIGHT = 10
    WIDTH = 10
    J = 1.0
    gamma = 0.1
    # Сколько самых медленных мод (собственных пар) мы хотим найти и сохранить
    NUM_MODES_TO_FIND = 4

    # 2. ПАРАМЕТРЫ ПАКЕТНОЙ ОБРАБОТКИ
    # Массив интересующих нас параметров p
    p_values =np.logspace(-4, 0, num=40)
    #p_values = [0.0, 0.01, 0.1, 0.5, 1.0]
    # Число запусков для каждого значения p
    num_runs_per_p = 30

    # 3. НАСТРОЙКА ФАЙЛА ДЛЯ СОХРАНЕНИЯ
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"rewiring_spectrum_data_{HEIGHT}x{WIDTH}_{timestamp}.npz"

    # Словарь для сбора всех данных перед сохранением
    data_to_save = {}

    print("=" * 60)
    print("            НАЧАЛО ПАКЕТНОГО АНАЛИЗА СПЕКТРОВ")
    print("=" * 60)
    print(f"Параметры решетки: {HEIGHT}x{WIDTH} (N={HEIGHT * WIDTH})")
    print(f"J={J}, gamma={gamma}")
    print(f"Будет найдено {NUM_MODES_TO_FIND} медленных мод.")
    print(f"Вероятности p для анализа: {p_values}")
    print(f"Запусков на каждую p: {num_runs_per_p}")
    print(f"Результаты будут сохранены в файл: {output_filename}\n")

    # Определение количества доступных ядер
    num_cores = os.cpu_count()
    print(f"Доступно ядер: {num_cores}. Параллелизация на {num_cores} процессах.\n")
    num_cores=num_cores//2
    # 4. ГЛАВНЫЙ ЦИКЛ С ПАРАЛЛЕЛИЗАЦИЕЙ
    total_runs = len(p_values) * num_runs_per_p
    completed_runs = 0

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for p in p_values:
            for run_idx in range(num_runs_per_p):
                futures.append(executor.submit(single_run, p, run_idx, HEIGHT, WIDTH, J, gamma, NUM_MODES_TO_FIND))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            data_to_save.update(result)
            completed_runs += 1
            elapsed = time.time() - start_time
            if completed_runs > 0:
                avg_time_per_run = elapsed / completed_runs
                remaining_runs = total_runs - completed_runs
                estimated_remaining = avg_time_per_run * remaining_runs
                percentage = (completed_runs / total_runs) * 100
                print(f"Прогресс: {completed_runs}/{total_runs} ({percentage:.2f}%), Оставшееся время: {estimated_remaining / 60:.2f} минут")

    # 5. ФИНАЛЬНОЕ СОХРАНЕНИЕ
    print("\n--- Завершение всех симуляций ---")
    if data_to_save:
        print(f"Сохранение {len(data_to_save) // 3} наборов данных в файл...")
        # Используем ** для распаковки словаря в именованные аргументы
        np.savez_compressed(output_filename, **data_to_save)
        print(f"Данные успешно сохранены в '{os.path.abspath(output_filename)}'")
    else:
        print("Нет данных для сохранения.")

    print("\n--- Скрипт завершил работу. ---")