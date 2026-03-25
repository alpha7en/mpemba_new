import numpy as np
import random
from collections import deque
from scipy.sparse import csc_matrix, identity, kron, diags
from scipy.sparse.linalg import eigs


# ==========================================
# 1. ТОПОЛОГИЯ (из main_simulation.py)
# ==========================================

def check_connectivity(tau: np.ndarray) -> bool:
    """Проверка связности (BFS)."""
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


def generate_rewired_tau_sparse(height: int, width: int, p: float) -> np.ndarray:
    """Генерация с гарантией связности."""
    while True:
        initial_tau = generate_grid_tau(height, width)
        tau = initial_tau.copy()
        N = height * width
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if initial_tau[i, j] == 1]
        random.shuffle(edges)

        for u, v in edges:
            if random.random() < p:
                if random.random() < 0.5:
                    node, partner = u, v
                else:
                    node, partner = v, u

                neighbors = np.where(tau[node, :] == 1)[0]
                forbidden = set(neighbors) | {node, partner}
                valid_targets = [n for n in range(N) if n not in forbidden]

                if valid_targets:
                    w = random.choice(valid_targets)
                    tau[u, v] = 0;
                    tau[v, u] = 0
                    tau[node, w] = 1;
                    tau[w, node] = 1

        if check_connectivity(tau):
            return tau


# ==========================================
# 2. ФИЗИКА: SPARSE LIOUVILLIAN
# ==========================================

def build_liouvillian_sparse(tau: np.ndarray, J: complex, gamma: float) -> csc_matrix:
    """Строит разреженный Лиувиллиан."""
    N = tau.shape[0]
    tau_sparse = csc_matrix(tau, dtype=np.complex128)

    H = -J / 2.0 * tau_sparse
    I = identity(N, dtype=np.complex128, format='csc')

    # L_H = -i [H, rho]
    L_H = -1j * (kron(I, H) - kron(H.T, I))

    diag_LD_vec = np.full(N * N, -gamma, dtype=np.complex128)
    indices = np.arange(N)
    # Диагональные элементы (населенности) не затухают
    diag_LD_vec[indices * N + indices] = 0.0
    L_D = diags(diag_LD_vec, 0, format='csc')

    return (L_H + L_D).asformat('csc')


# ==========================================
# 3. АНАЛИЗ: БИОРТОГОНАЛЬНЫЙ БАЗИС (SPARSE)
# ==========================================

def get_biorthogonal_modes_sparse_strict(L: csc_matrix, num_modes: int):
    """
    Находит правые (v) и левые (w) собственные векторы для k мод.
    Использует eigs для L и L^dagger.
    """
    sigma = 1e-9  # Сдвиг для поиска значений около 0

    # 1. Находим ПРАВЫЕ векторы (L v = lambda v)
    try:
        vals_R, vecs_R = eigs(L, k=num_modes, sigma=sigma, which='LM')
    except:
        return None, None, None

    # Сортируем правые по Re(lambda)
    sort_idx = np.argsort(vals_R.real)[::-1]
    vals_R = vals_R[sort_idx]
    vecs_R = vecs_R[:, sort_idx]

    # 2. Находим ЛЕВЫЕ векторы (L^dagger w = lambda^* w)
    # L.H - эрмитово сопряжение
    try:
        vals_L, vecs_L = eigs(L.H, k=num_modes, sigma=np.conj(sigma), which='LM')
    except:
        return None, None, None

    # 3. Сопоставление (Matching) левых и правых векторов
    # eigs может вернуть их в разном порядке. Сопоставляем по собственным числам.
    final_w = np.zeros_like(vecs_R)

    for i, val_r in enumerate(vals_R):
        # Ищем в vals_L значение, которое ближе всего к conj(val_r)
        dist = np.abs(vals_L - np.conj(val_r))
        best_match_idx = np.argmin(dist)
        final_w[:, i] = vecs_L[:, best_match_idx]

    return vals_R, vecs_R, final_w


# ==========================================
# 4. ВЫЧИСЛЕНИЕ ВОЗБУДИМОСТИ (ВЕКТОРИЗОВАНО)
# ==========================================

def calculate_excitability_map_vectorized(left_vecs, right_vecs, k_idx, N):
    """
    Считает B(k, i) строго по формуле, но без цикла Python.

    Теория:
    B(k, i) = | <w_k | rho_i> / <w_k | v_k> |^2
    где rho_i = |i><i|. Вектор vec(rho_i) имеет единицу на позиции (i*N + i).

    Скалярное произведение <w_k | vec(rho_i)> берет (i*N+i)-ю компоненту вектора w_k (сопряженную).
    """
    w_k = left_vecs[:, k_idx]
    v_k = right_vecs[:, k_idx]

    # Знаменатель (норма)
    norm = np.vdot(w_k, v_k)
    if np.isclose(norm, 0): return np.zeros(N)

    # Числитель: извлекаем компоненты w_k, соответствующие диагональным элементам матрицы плотности.
    # Индексы диагонали при flatten('F'): 0, N+1, 2N+2...
    diag_indices = np.arange(N) * (N + 1)

    # w_k[idx].conj() эквивалентно <w_k | e_idx>
    projections = w_k[diag_indices].conj()

    # Итоговая формула
    B_map = np.abs(projections / norm) ** 2

    return B_map