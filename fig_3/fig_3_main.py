import numpy as np

# --- Генератор топологии ---
def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Generates adjacency matrix for a 2D lattice with nearest neighbors."""
    N = height * width
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        # Connection with the right neighbor (if not on the right edge)
        if (i + 1) % width != 0:
            tau[i, i + 1] = 1
            tau[i + 1, i] = 1
        # Connection with the bottom neighbor (if not on the bottom edge)
        if i < N - width:
            tau[i, i + width] = 1
            tau[i + width, i] = 1
    return tau

# --- Генераторы начальных состояний ---
def _create_mixed_state_rho(size: int, indices: list) -> np.ndarray:
    """Helper function to create a MIXED state."""
    num_indices = len(indices)
    if num_indices == 0: return np.zeros((size, size), dtype=np.complex128)
    rho = np.zeros((size, size), dtype=np.complex128)
    for idx in indices:
        rho[idx, idx] = 1.0 / num_indices
    return rho

def _create_pure_state_rho(size: int, indices: list) -> np.ndarray:
    """Helper function to create a PURE (entangled) state."""
    num_indices = len(indices)
    if num_indices == 0: return np.zeros((size, size), dtype=np.complex128)
    psi = np.zeros(size, dtype=np.complex128)
    psi[indices] = 1.0 / np.sqrt(num_indices)
    return np.outer(psi, psi.conj())

def create_localized_state(h, w): return _create_mixed_state_rho(h*w, [(h//2)*w + (w//2)])
def create_opposite_corners_state(h, w): return _create_mixed_state_rho(h*w, [0, h*w-1])
def create_four_corners_state(h, w): return _create_mixed_state_rho(h*w, [0, w-1, (h-1)*w, h*w-1])
def create_mixed_diagonal_state(h, w): return _create_mixed_state_rho(h*w, [i*w+i for i in range(min(h,w))])
def create_entangled_diagonal_state(h, w): return _create_pure_state_rho(h*w, [i*w+i for i in range(min(h,w))])
def create_inner_corners_state(h, w):
    if h < 3 or w < 3: return None
    return _create_mixed_state_rho(h*w, [1*w+1, 1*w+(w-2), (h-2)*w+1, (h-2)*w+(w-2)])
def create_top_bottom_edges_state(h, w):
    top = list(range(w)); bottom = list(range((h-1)*w, h*w))
    return _create_mixed_state_rho(h*w, sorted(list(set(top + bottom))))
def create_checkerboard_state(h, w):
    return _create_mixed_state_rho(h*w, [i*w+j for i in range(h) for j in range(w) if (i+j)%2==0])
def create_boundary_state(h, w):
    if h<3 or w<3: return None
    top=list(range(w)); bottom=list(range((h-1)*w,h*w))
    left=[i*w for i in range(1,h-1)]; right=[i*w+w-1 for i in range(1,h-1)]
    return _create_mixed_state_rho(h*w, sorted(list(set(top+bottom+left+right))))



import matplotlib.pyplot as plt
# Предполагается, что файл fully_simulation_core.py находится в той же директории
# или установлен как библиотека.
from fully_simulation_core import QuantumSimulatorCore

# --- 1. Настройка эксперимента ---
# Создаем экземпляр симулятора
QScore = QuantumSimulatorCore()

# Физические параметры
J_val = 1.0
gamma_val = 0.5  # Как в вашем примере

# Словарь с генераторами начальных состояний
initial_state_generators = {
    "1. Center (Mixed)": create_localized_state,
    "2. Opposite corners (Mixed)": create_opposite_corners_state,
    "3. Four corners (Mixed)": create_four_corners_state,
    "4. Mixed diagonal": create_mixed_diagonal_state,
    "5. Entangled diagonal": create_entangled_diagonal_state,
    "6. Inner corners (Mixed)": create_inner_corners_state,
    "7. Edges (top/bottom) (Mixed)": create_top_bottom_edges_state,
    "8. Checkerboard (Mixed)": create_checkerboard_state,
    "9. Boundary (Mixed)": create_boundary_state,
}

# --- 2. Основной цикл исследования ---
for n in range(10, 11):
    print("\n" + "#" * 70)
    print(f"# STARTING SIMULATIONS FOR LATTICE {n}x{n} (N={n * n})")
    print("#" * 70)

    # --- Шаг А: Подготовка к симуляции для данного N ---

    # Создаем фигуру и оси для графика
    fig, ax = plt.subplots(figsize=(12, 8))

    # Генерируем топологию
    tau = generate_grid_tau(n, n)

    # Строим Лиувиллиан (один раз для всех симуляций с данным N)
    print("Building Liouvillian...")
    L = QScore.build_liouvillian(tau, J_val, gamma_val)
    print("Liouvillian built.")

    # --- Шаг Б: Цикл по начальным состояниям и их симуляция ---
    case_data = {}  # Для хранения данных для случаев 3, 4, 5
    for name, generator in initial_state_generators.items():
        print(f"\n--- Simulation for state: '{name}' ---")

        rho_initial = generator(n, n)

        if rho_initial is None:
            print("State not defined for this lattice size. Skipping.")
            continue

        # ВАЖНО: Нормировка матрицы плотности
        trace = np.trace(rho_initial)
        if np.isclose(trace, 0):
            print("Density matrix is zero. Skipping.")
            continue
        rho_initial_normalized = rho_initial / trace

        # Запускаем симуляцию
        if name != "8. Checkerboard (Mixed)!" and name != "9. Boundary (Mixed)!":
            t, D, elapsed = QScore.run_simulation(L, rho_initial_normalized)
        else:
            pass
            # t, D, elapsed = QScore.run_simulation(L, rho_initial_normalized, method="BDF")

        print(f"Simulation completed in {elapsed:.4f} s. ({len(t)} steps)")

        # Добавляем линию на общий график
        if name.startswith("3. Four corners") or name.startswith("4. Mixed diagonal") or name.startswith("5. Entangled diagonal"):
            ax.plot(t, D, label=name, lw=3)  # Более толстые линии
            case_data[name] = (t, D)  # Сохраняем данные для пересечений
        else:
            ax.plot(t, D, label=name, lw=1, alpha=0.5)  # Блеклые линии

    # Проверка пересечений для случаев 3, 4, 5
    def find_intersection(t1, D1, t2, D2):
        for i in range(1, len(t1)):
            if (D1[i - 1] - D2[i - 1]) * (D1[i] - D2[i]) < 0:  # Проверка смены знака
                intersect_t = t1[i - 1] + (t1[i] - t1[i - 1]) * abs(D1[i - 1] - D2[i - 1]) / abs((D1[i] - D2[i]) - (D1[i - 1] - D2[i - 1]))
                intersect_D = D1[i - 1] + (D1[i] - D1[i - 1]) * (intersect_t - t1[i - 1]) / (t1[i] - t1[i - 1])
                return intersect_t, intersect_D
        return None

    if "3. Four corners (Mixed)" in case_data and "5. Entangled diagonal" in case_data:
        t3, D3 = case_data["3. Four corners (Mixed)"]
        t5, D5 = case_data["5. Entangled diagonal"]
        intersection = find_intersection(t3, D3, t5, D5)
        if intersection:
            ax.plot(intersection[0], intersection[1], 'o', markersize=8, color='red')  # Жирная точка

    if "5. Entangled diagonal" in case_data and "4. Mixed diagonal" in case_data:
        t5, D5 = case_data["5. Entangled diagonal"]
        t4, D4 = case_data["4. Mixed diagonal"]
        intersection = find_intersection(t5, D5, t4, D4)
        if intersection:
            ax.plot(intersection[0], intersection[1], 'o', markersize=8, color='red')  # Жирная точка

    # --- Шаг В: Финализация и отображение графика для данного N ---
    ax.set_title(f"Relaxation dynamics for {n}x{n} lattice", fontsize=16)
    ax.set_xlabel("Time, t (s)", fontsize=12)
    ax.set_ylabel("Distance metric D(t)", fontsize=12)
    ax.grid(True, which='both', linestyle='--')
    ax.legend(loc='upper right', fontsize=10)


    print(f"\nDisplaying plot for lattice {n}x{n}...")
    plt.tight_layout()
    plt.savefig(f"9states_modes_simulation_line_{n}x{n}_with_selection_ENG.png", bbox_inches='tight', pad_inches=0.05)

    plt.show()