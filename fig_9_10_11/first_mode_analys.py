from reconecting.bright_dark.bright_dark_math import *
import numpy as np
from numba import njit


def get_projection_coefficient(left_vec, right_vec, state_vec):
    """
    Вычисляет коэффициент разложения c_k = <w_k | rho> / <w_k | v_k>
    с учетом биортогональности.
    """

    numerator = np.vdot(left_vec, state_vec)
    denominator = np.vdot(left_vec, right_vec)

    if np.abs(denominator) < 1e-12:
        return 0.0

    return numerator / denominator


def entropic_distance(rho_vec, N):
    """
    Вычисляет расстояние до равновесия (метрика на основе энтропии).
    D = log(N) - S(rho).
    """

    rho = rho_vec.reshape((N, N))

    evals = np.linalg.eigvalsh(rho)

    # Фильтрация шумов и нормировка
    evals = evals[evals > 1e-15]
    evals = evals / np.sum(evals)

    entropy = -np.sum(evals * np.log(evals))
    max_entropy = np.log(N)

    return max_entropy - entropy


@njit(fastmath=True)
def fast_diag_entropy(rho_vec, N):
    # Извлекаем диагональ (населенности)
    # rho_vec хранит матрицу построчно. Диагональные элементы: 0, N+1, 2(N+1)...
    diag_indices = np.arange(N) * (N + 1)
    probs = rho_vec[diag_indices].real

    # Фильтрация нулей и нормировка (на всякий случай)
    probs = probs[probs > 1e-15]
    probs /= np.sum(probs)

    entropy = -np.sum(probs * np.log(probs))
    return np.log(N) - entropy


def find_guaranteed_mpemba_dense(left_vecs, right_vecs, N, mode_idx=1, threshold_ratio=5.0):
    """
    Автоматически подбирает состояния Hot и Cold на основе карт возбудимости
    из core_dense.
    """
    print(f"--- Searching for Mpemba states (Mode k={mode_idx}) ---")

    b_map = calculate_excitability_map_dense(left_vecs, right_vecs, mode_idx, N)

    w_vec = left_vecs[:, mode_idx]
    v_vec = right_vecs[:, mode_idx]

    dark_node = np.argmin(b_map)

    rho_hot_mat = np.zeros((N, N), dtype=complex)
    rho_hot_mat[dark_node, dark_node] = 1.0
    vec_hot = rho_hot_mat.flatten()  # order='C' по умолчанию

    c1_hot = np.abs(get_projection_coefficient(w_vec, v_vec, vec_hot))
    dist_hot = entropic_distance(vec_hot, N)

    sorted_indices = np.argsort(b_map)[::-1]  # От самых ярких к темным

    best_vec_cold = None
    best_vec_cold_score = None
    best_ratio = -1.0
    best_M = 0

    best_score = -1.0
    best_M_score = 0

    # Перебираем размер смеси M от 2 до N/2
    for M in range(2, max(3, int(N / 2))):
        top_nodes = sorted_indices[:M]

        rho_cold_mat = np.zeros((N, N), dtype=complex)
        for node in top_nodes:
            rho_cold_mat[node, node] = 1.0 / M
        vec_cold = rho_cold_mat.flatten()

        c1_cold = np.abs(get_projection_coefficient(w_vec, v_vec, vec_cold))
        dist_cold = entropic_distance(vec_cold, N)

        if dist_cold >= dist_hot - 0.01:
            continue

        ratio = (c1_cold / (c1_hot + 1e-15)) ** 2
        dist_diff = np.abs(dist_cold - dist_hot)  # Тот самый |D_cold - D_hot|

        current_score = ratio * dist_diff
        print(M, ratio, current_score)

        if ratio > best_ratio:
            best_ratio = ratio
            best_vec_cold = vec_cold
            best_M = M

        if current_score > best_score:
            best_score = current_score
            best_vec_cold_score = vec_cold
            best_M_score = M

    print(f"Found States based on ratio:")
    print(f"  HOT Node: {dark_node} (Brightness: {b_map[dark_node]:.2e})")
    print(f"  COLD Mix Size: {best_M} nodes")
    print(f"  Ratio |c_cold|/|c_hot|: {best_ratio:.2f}")

    print(f"Found States based on score:")
    print(f"  HOT Node: {dark_node} (Brightness: {b_map[dark_node]:.2e})")
    print(f"  COLD Mix Size: {best_M_score} nodes")
    print(f"  Ratio |c_cold|/|c_hot|: {best_score:.2f}")

    if best_ratio > threshold_ratio:
        print("  >> GUARANTEE: High projection ratio predicts Mpemba effect.")
    else:
        print("  >> WARNING: Ratio is low. Effect might be weak.")
        # Если не нашли хорошего Cold (например, граф слишком симметричный),
        # возвращаем хотя бы что-то валидное
        if best_vec_cold is None:
            rho_def = np.zeros((N, N), dtype=complex);
            rho_def[sorted_indices[0], sorted_indices[0]] = 1.0
            best_vec_cold = rho_def.flatten()

    return vec_hot, best_vec_cold, b_map, best_vec_cold_score


def run_experiment_dense(height=5, width=5, p=0.15, J=1.0, gamma=0.5):
    N = height * width

    tau = generate_rewired_tau_dense(height, width, p)

    L = build_liouvillian_dense(tau, J, gamma)

    evals, left_vecs, right_vecs = analyze_modes_dense_strict(L)

    slowest_idx = 1
    lambda_slow = evals[slowest_idx]
    print(f"Slowest Mode lambda: {lambda_slow:.4f} (Real part: {lambda_slow.real:.4f})")

    vec_hot, vec_cold, b_map, vec_cold_score = find_guaranteed_mpemba_dense(
        left_vecs, right_vecs, N, mode_idx=slowest_idx
    )
    return tau, L, vec_hot, vec_cold, height, width


if __name__ == "__main__":
    run_experiment_dense()
