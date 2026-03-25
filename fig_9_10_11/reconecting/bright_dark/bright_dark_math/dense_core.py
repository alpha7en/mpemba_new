import numpy as np
import random
from collections import deque
from scipy.linalg import eig

# ==============================================================================
# БЛОК 1: ТОПОЛОГИЯ
# Источник: main_simulation.py
# Логика: Генерация Ваттса-Строгаца с ГАРАНТИЕЙ связности (цикл while + BFS)
# ==============================================================================

def check_connectivity(tau: np.ndarray) -> bool:
    """Проверка связности графа (BFS)."""
    N = tau.shape[0]
    if N == 0: return True
    visited = set()
    q = deque([0])
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
    return count == N

def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Базовая решетка."""
    N = height * width
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        if (i + 1) % width != 0: tau[i, i + 1] = 1; tau[i + 1, i] = 1
        if i < N - width: tau[i, i + width] = 1; tau[i + width, i] = 1
    return tau

def generate_rewired_tau_dense(height: int, width: int, p: float) -> np.ndarray:
    """
    Генерация сети для плотных матриц (возвращает numpy array).
    Логика идентична main_simulation.py.
    """
    while True:
        initial_tau = generate_grid_tau(height, width)
        tau = initial_tau.copy()
        N = height * width
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if initial_tau[i, j] == 1]
        random.shuffle(edges)

        for u, v in edges:
            if random.random() < p:
                # Выбираем, какой конец отсоединять
                if random.random() < 0.5:
                    node_to_rewire, original_partner = u, v
                else:
                    node_to_rewire, original_partner = v, u

                neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
                forbidden = set(neighbors) | {node_to_rewire, original_partner}
                valid_targets = [n for n in range(N) if n not in forbidden]

                if valid_targets:
                    w = random.choice(valid_targets)
                    tau[u, v] = 0; tau[v, u] = 0
                    tau[node_to_rewire, w] = 1; tau[w, node_to_rewire] = 1

        if check_connectivity(tau):
            return tau

# ==============================================================================
# БЛОК 2: ФИЗИКА (DENSE)
# Источник: core.py (старый, проверенный скрипт)
# Логика: np.kron, scipy.linalg.eig, flatten('F')
# ==============================================================================

def build_liouvillian_dense(tau, J, gamma):
    """
    Строит плотный Лиувиллиан.
    Источник: core.py
    """
    tau = np.asarray(tau, dtype=np.complex128)
    N = tau.shape[0]
    H = -J / 2.0 * tau
    I = np.eye(N, dtype=np.complex128)

    # L_H = -i [H, rho] = -i(I x H - H.T x I)
    # Используем np.kron (плотный)
    L_H = -1j * (np.kron(I, H) - np.kron(H.T, I))

    diag_LD = np.full(N * N, -gamma, dtype=np.complex128)
    for i in range(N):
        # Элементы на диагонали матрицы плотности (населенности) не затухают от чистой дефазировки
        # Индекс (i, i) в векторе - это i * N + i (при flatten order='C')
        # ИЛИ i + i * N (при flatten order='F').
        # В core.py использовался order='F' для векторизации, но здесь диагональная матрица.
        # Внимание: np.diag создает диагональную матрицу.
        # Для диагонали (i,i) индекс в векторе всегда i*(N+1), независимо от F или C,
        # так как матрица квадратная.
        diag_LD[i * (N + 1)] = 0.0

    L_D = np.diag(diag_LD)

    return np.add(L_H, L_D)

def analyze_modes_dense_strict(L):
    """
    Находит левые и правые вектора.
    Источник: core.py (analyze_liouvillian_modes)
    """
    # left=True ОБЯЗАТЕЛЬНО для корректной проекции в неэрмитовой системе
    eigenvalues, left_vecs, right_vecs = eig(L, left=True, right=True)

    # Сортировка по убыванию Re(lambda) (медленные в начале)
    sort_indices = np.argsort(eigenvalues.real)[::-1]

    return (eigenvalues[sort_indices],
            left_vecs[:, sort_indices],
            right_vecs[:, sort_indices])

def calculate_excitability_map_dense(left_vecs, right_vecs, k_idx, N):
    """
    Вычисляет карту B(k, i) = |c_k|^2 для локальных состояний.
    Использует строгую формулу проекции из переписки.

    B(k, i) = | <w_k | rho_i> / <w_k | v_k> |^2
    """
    w_k = left_vecs[:, k_idx]
    v_k = right_vecs[:, k_idx]

    # 1. Нормировка на скалярное произведение <w_k | v_k>
    # np.vdot(a, b) возвращает a.conj() * b
    norm_factor = np.vdot(w_k, v_k)

    if np.isclose(norm_factor, 0):
        return np.zeros(N)

    # 2. Проекция на локальные состояния
    # rho_i = |i><i|.
    # При векторизации vec(rho_i) это вектор, где 1 стоит только на позиции,
    # соответствующей диагональному элементу (i, i).
    # Индекс этого элемента (при flatten 'F', как в core.py): row + col*N = i + i*N.
    diag_indices = np.arange(N) * (N + 1)

    # Скалярное произведение <w_k | vec(rho_i)> равно (i+i*N)-му элементу w_k,
    # взятому с сопряжением (из-за определения скалярного произведения в гильбертовом пр-ве).
    # w_k[idx].conj() * 1.0
    projections = w_k[diag_indices].conj()

    # 3. Итоговая формула
    B_map = np.abs(projections / norm_factor)**2

    return B_map