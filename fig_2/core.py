

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, logm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
from fully_simulation_core import QuantumSimulatorCore
QScore=QuantumSimulatorCore()

# --- Функции ядра до визуализации (без изменений) ---
def build_liouvillian(tau, J, gamma):  # ...
    tau = np.asarray(tau, dtype=np.complex128);
    N = tau.shape[0];
    H = -J / 2.0 * tau;
    I = np.eye(N, dtype=np.complex128)
    L_H = -1j * (np.kron(I, H) - np.kron(H.T, I));
    diag_LD = np.full(N * N, -gamma, dtype=np.complex128)
    for i in range(N): diag_LD[i * N + i] = 0.0
    L_D = np.diag(diag_LD);
    return H, np.add(L_H, L_D)


def analyze_liouvillian_modes(L):  # ...
    eigenvalues, left_eigenvectors, right_eigenvectors = eig(L, left=True, right=True)
    sort_indices = np.argsort(eigenvalues.real)[::-1]
    return (eigenvalues[sort_indices], left_eigenvectors[:, sort_indices], right_eigenvectors[:, sort_indices])


def project_rho_on_modes(rho_initial, left_vecs, right_vecs):  # ...
    N = rho_initial.shape[0];
    rho_vec = rho_initial.flatten('F');
    num_modes = N * N
    coefficients = np.zeros(num_modes, dtype=np.complex128)
    for k in range(num_modes):
        w_k, v_k = left_vecs[:, k], right_vecs[:, k]
        norm_factor = np.vdot(w_k, v_k);
        projection = np.vdot(w_k, rho_vec)
        coefficients[k] = projection / norm_factor if not np.isclose(norm_factor, 0) else 0
    return coefficients


def calculate_distance_metric(rho):  # ...
    N = rho.shape[0];
    if np.isclose(np.trace(np.dot(rho, rho)), 1.0): return np.log(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning);
        log_rho = logm(rho)
    tr_rho_log_rho = np.trace(np.dot(rho, log_rho));
    return np.log(N) + tr_rho_log_rho.real


def generate_grid_tau(height, width):  # ...
    N = height * width;
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        if (i + 1) % width != 0: tau[i, i + 1] = 1; tau[i + 1, i] = 1
        if i < N - width: tau[i, i + width] = 1; tau[i + width, i] = 1
    return tau


def _create_mixed_state_rho(size, indices):  # ...
    num_indices = len(indices);
    if num_indices == 0: return np.zeros((size, size)), []
    rho = np.zeros((size, size), dtype=np.complex128)
    for idx in indices: rho[idx, idx] = 1.0 / num_indices
    return rho, indices


def _create_pure_state_rho(size, indices):  # ...
    num_indices = len(indices);
    if num_indices == 0: return np.zeros((size, size)), []
    psi = np.zeros(size, dtype=np.complex128);
    psi[indices] = 1.0 / np.sqrt(num_indices)
    return np.outer(psi, psi.conj()), indices


# --- Генераторы состояний (без изменений) ---
def create_localized_state(h, w): return _create_mixed_state_rho(h * w, [(h // 2) * w + (w // 2)])


def create_opposite_corners_state(h, w): return _create_mixed_state_rho(h * w, [0, h * w - 1])


def create_four_corners_state(h, w): return _create_mixed_state_rho(h * w, [0, w - 1, (h - 1) * w, h * w - 1])


def create_mixed_diagonal_state(h, w): return _create_mixed_state_rho(h * w, [i * w + i for i in range(min(h, w))])


def create_entangled_diagonal_state(h, w): return _create_pure_state_rho(h * w, [i * w + i for i in range(min(h, w))])


def create_inner_corners_state(h, w): return _create_mixed_state_rho(h * w,
                                                                     [1 * w + 1, 1 * w + (w - 2), (h - 2) * w + 1,
                                                                      (h - 2) * w + (w - 2)]) if h > 2 and w > 2 else (
None, None)


def create_top_bottom_edges_state(h, w):
    top = list(range(w));
    bottom = list(range((h - 1) * w, h * w))
    return _create_mixed_state_rho(h * w, sorted(list(set(top + bottom))))


def create_checkerboard_state(h, w): return _create_mixed_state_rho(h * w,
                                                                    [i * w + j for i in range(h) for j in range(w) if
                                                                     (i + j) % 2 == 0])


def create_boundary_state(h, w):
    if h < 3 or w < 3: return (None, None)
    top = list(range(w));
    bottom = list(range((h - 1) * w, h * w));
    left = [i * w for i in range(1, h - 1)];
    right = [i * w + w - 1 for i in range(1, h - 1)]
    return _create_mixed_state_rho(h * w, sorted(list(set(top + bottom + left + right))))


# --- НОВЫЙ БЛОК: Функции визуализации мод ---

def draw_single_mode_on_axis(ax, population_map, height, width, k_scaler, title_text):
    """Рисует одну диаграмму моды на предоставленной оси Matplotlib (ax)."""
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    for y in range(height):
        for x in range(width):
            value = population_map[y, x]
            color = 'white'
            if value > 1e-6: color = 'red'
            elif value < -1e-6: color = 'blue'
            alpha = min(1.0, np.abs(value) * k_scaler)
            circle = plt.Circle((x, y), radius=0.35, facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)

    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-0.6, height - 0.4)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(title_text, fontsize=9, y=-0.3)


# --- ОБНОВЛЕННАЯ ВЕРСИЯ ЯДРА ВИЗУАЛИЗАЦИИ ---

def analyze_projections_and_get_plot_range(coefficients: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Анализирует коэффициенты проекций.
    Возвращает:
    - normalized_contributions: Нормированные вклады мод (в долях).
    - K_max: Максимальный индекс моды для отображения на гистограмме.
    - top_3_indices: Индексы трех мод с наибольшим вкладом.
    """
    contributions = np.abs(coefficients[1:]) ** 2
    total_contribution = np.sum(contributions)
    if total_contribution < 1e-12:
        return np.zeros_like(contributions), 8, np.array([], dtype=int)

    normalized_contributions = contributions / total_contribution
    mode_indices = np.arange(1, len(normalized_contributions) + 1)
    sorted_mode_indices_by_contrib = mode_indices[np.argsort(normalized_contributions)[::-1]]
    top_3_indices = sorted_mode_indices_by_contrib[:3]
    top_8_indices = sorted_mode_indices_by_contrib[:8]
    K_max = np.max(top_8_indices) if len(top_8_indices) > 0 else 8

    return normalized_contributions, K_max, top_3_indices


def visualize_experiment_for_n(n, J, gamma, state_generators):
    N = n * n
    tau = generate_grid_tau(n, n)
    _, L = build_liouvillian(tau, J, gamma)
    _, left_vecs, right_vecs = analyze_liouvillian_modes(L)

    # Шаг 1: Предварительный расчет
    results_data = []
    for name, generator in state_generators.items():
        rho, indices = generator(n, n)
        if rho is None: continue
        metric = calculate_distance_metric(rho)
        results_data.append({'name': name, 'rho': rho, 'indices': indices, 'metric': metric})
    results_data.sort(key=lambda x: x['metric'], reverse=True)
    num_states = len(results_data)

    # Шаг 2: Расчет проекций и глобальных параметров (ЛОГИКА НЕ МЕНЯЛАСЬ)
    global_max_contrib = 0
    max_abs_mode_pop = 0.0
    for data in results_data:
        coeffs = project_rho_on_modes(data['rho'], left_vecs, right_vecs)
        contribs, K_max, top_3 = analyze_projections_and_get_plot_range(coeffs)
        data.update({'contribs': contribs, 'K_max': K_max, 'top_3': top_3})
        if len(contribs) > 0: global_max_contrib = max(global_max_contrib, contribs.max())
        for k in top_3:
            v_k = right_vecs[:, k]
            rho_k = v_k.reshape((N, N), order='F')
            pop_k = np.diag(rho_k).real
            current_max = np.max(np.abs(pop_k))
            if current_max > max_abs_mode_pop: max_abs_mode_pop = current_max

    k_scaler = 1.0 / (max_abs_mode_pop + 1e-9)

    # Шаг 3: Основной цикл отрисовки
    fig = plt.figure(figsize=(12, 5 * num_states))
    
    outer_gs = gridspec.GridSpec(num_states, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.3)

    for i, data in enumerate(results_data):
        # --- Левая ячейка: Схема состояния (ЛОГИКА НЕ МЕНЯЛАСЬ) ---
        ax_rho = fig.add_subplot(outer_gs[i, 0])
        ax_rho.set_title(data['name'], fontsize=11)
        ax_rho.set_xticks(np.arange(n)); ax_rho.set_yticks(np.arange(n))
        ax_rho.set_xticklabels([]); ax_rho.set_yticklabels([])
        schematic = np.zeros((n, n))
        val = 1.0 / len(data['indices']) if data['indices'] else 0
        for idx in data['indices']:
            row, col = divmod(idx, n)
            schematic[row, col] = val
        ax_rho.imshow(schematic, cmap='viridis', vmin=0, vmax=max(0.001, schematic.max()), aspect='equal')

        # --- Правая ячейка: Гистограмма и моды ---
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i, 1], height_ratios=[1, 1], hspace=0.6)

        # Верхняя часть правой ячейки: Метрика и гистограмма
        bar_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_gs[0], height_ratios=[0.15, 1], hspace=0.3)
        ax_text = fig.add_subplot(bar_gs[0]); ax_text.axis('off')
        ax_text.text(0.5, 0.5, f"Метрика D(ρ(0)) = {data['metric']:.4f}", ha='center', va='center', fontsize=12)
        
        ax_bar = fig.add_subplot(bar_gs[1])
        
        # --- ИЗМЕНЕНИЯ ТОЛЬКО ЗДЕСЬ ---
        ax_bar.set_yscale('log')
        ymin = 0.01 
        formatter = FuncFormatter(lambda y, _: f'{y:g}%')
        ax_bar.yaxis.set_major_formatter(formatter)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        contribs, local_K_max = data['contribs'], data['K_max']
        modes_to_plot = np.arange(1, local_K_max + 1)
        contribs_to_plot = contribs[:local_K_max] * 100
        
        # Отфильтровываем нулевые значения для корректной работы log-шкалы
        valid_indices = np.where(contribs_to_plot > ymin)[0]

        if local_K_max <= 40:
            if len(valid_indices) > 0:
                ax_bar.bar(modes_to_plot[valid_indices], contribs_to_plot[valid_indices], color='skyblue', width=0.8)
            ax_bar.set_xticks(modes_to_plot)
            ax_bar.set_xticklabels([f"{k}" for k in modes_to_plot], rotation=90, fontsize=8)
        else:
            if len(valid_indices) > 0:
                ax_bar.vlines(modes_to_plot[valid_indices], ymin, contribs_to_plot[valid_indices], color='skyblue')
            tick_step = int(np.ceil(local_K_max / 20))
            ticks = np.arange(1, local_K_max + 1, tick_step)
            ax_bar.set_xticks(ticks)
            ax_bar.set_xticklabels([f"{k}" for k in ticks], rotation=90, fontsize=8)

        ax_bar.set_xlabel("Индекс моды k (от 1)")
        ax_bar.set_ylabel("Относительный вклад, % (log)")
        ax_bar.set_xlim(0.5, local_K_max + 0.5)
        
        ymax = global_max_contrib * 100 * 1.15 if global_max_contrib > 0 else 1
        ax_bar.set_ylim(bottom=ymin, top=max(ymax, ymin * 10))

        # --- Нижняя часть правой ячейки: Три диаграммы мод (ЛОГИКА НЕ МЕНЯЛАСЬ) ---
        bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=right_gs[1], wspace=0.1)
        top_indices = data['top_3']
        if len(top_indices) > 0:
            for j, k in enumerate(top_indices):
                ax_mode = fig.add_subplot(bottom_gs[j])
                v_k = right_vecs[:, k]
                rho_k = v_k.reshape((N, N), order='F')
                population_map = np.diag(rho_k).real.reshape((n, n))
                contribution_percent = data['contribs'][k-1] * 100
                title = f"Мода k={k}\n({contribution_percent:.1f}%)"
                draw_single_mode_on_axis(ax_mode, population_map, n, n, k_scaler, title)
        for j in range(len(top_indices), 3):
            fig.add_subplot(bottom_gs[j]).axis('off')

    plt.tight_layout()
    plt.savefig(f"9states_modes_bar_chart_with_modes_{n}x{n}_log_afterdebag_gamma01.png", bbox_inches='tight', pad_inches=0.05)
    plt.show()
