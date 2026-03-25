import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import math
from scipy.linalg import eig


# ==============================================================================
# --- БЛОК 1: ВЫЧИСЛИТЕЛЬНЫЕ ФУНКЦИИ ЯДРА ---
# ==============================================================================

def build_liouvillian(tau, J, gamma):
    tau = np.asarray(tau, dtype=np.complex128)
    N = tau.shape[0]
    H = -J / 2.0 * tau
    I = np.eye(N, dtype=np.complex128)
    L_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
    diag_LD = np.full(N * N, -gamma, dtype=np.complex128)
    for i in range(N):
        diag_LD[i * N + i] = 0.0
    L_D = np.diag(diag_LD)
    L = L_H + L_D
    return H, L


def analyze_liouvillian_modes(L):
    eigenvalues, left_eigenvectors, right_eigenvectors = eig(L, left=True, right=True)
    sort_indices = np.argsort(eigenvalues.real)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_left_eigenvectors = left_eigenvectors[:, sort_indices]
    sorted_right_eigenvectors = right_eigenvectors[:, sort_indices]
    return sorted_eigenvalues, sorted_left_eigenvectors, sorted_right_eigenvectors


def generate_grid_tau(height: int, width: int) -> np.ndarray:
    N = height * width
    tau = np.zeros((N, N), dtype=int)
    if height == 1 or width == 1:
        for i in range(N - 1):
            tau[i, i + 1] = 1
            tau[i + 1, i] = 1
        return tau
    for i in range(N):
        if (i + 1) % width != 0:
            tau[i, i + 1] = 1
            tau[i + 1, i] = 1
        if i < N - width:
            tau[i, i + width] = 1
            tau[i + width, i] = 1
    return tau


# ==============================================================================
# --- БЛОК 2: ФУНКЦИИ ВИЗУАЛИЗАЦИИ ---
# (Финальная версия с правильным объединением ячеек и плотной компоновкой)
# ==============================================================================

def format_cell_value(value: float) -> str:
    """Форматирует значение до сотых, убирая ведущий ноль (0.01 -> .01)."""
    s = f"{value:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def draw_single_mode_on_axis(ax, population_map, height, width, k_scaler, original_k, lambda_k):
    """Рисует одну диаграмму моды, включая k и Im(lambda) в заголовок."""
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    for y in range(height):
        for x in range(width):
            value = population_map[y, x]
            color = 'white'
            if value > 1e-6:
                color = 'red'
            elif value < -1e-6:
                color = 'blue'

            alpha = min(1.0, np.abs(value) * k_scaler)
            circle = plt.Circle((x, y), radius=0.45, facecolor=color, alpha=alpha)
            ax.add_patch(circle)
            # ax.text(x, y, format_cell_value(value), ha='center', va='center', color='black', fontsize=9)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    title_text = f"k = {original_k}   Im(λ) ≈ {lambda_k.imag:.4f}"
    ax.set_title(title_text, fontsize=15, fontweight='bold')


def plot_dynamically_grouped_diagrams(
        sorted_eigenvalues: np.ndarray,
        sorted_eigenvectors: np.ndarray,
        height: int,
        width: int,
        num_modes_to_show: int
):
    """
    Создает финальное сверхплотное изображение, правильно объединяя ячейки
    для неполных групп.
    """
    N = height * width
    modes_to_process = range(1, min(num_modes_to_show, N * N))

    if not modes_to_process:
        print("Нет мод для визуализации (кроме k=0).")
        return

    # 1. ДИНАМИЧЕСКАЯ ГРУППИРОВКА
    groups = []
    if modes_to_process:
        current_group = [modes_to_process[0]]
        for i in range(1, len(modes_to_process)):
            k_curr = modes_to_process[i]
            k_prev = current_group[-1]
            if np.isclose(sorted_eigenvalues[k_curr].real, sorted_eigenvalues[k_prev].real) and len(current_group) < 4:
                current_group.append(k_curr)
            else:
                groups.append(current_group)
                current_group = [k_curr]
        groups.append(current_group)

    print(f"Найдено {len(groups)} групп мод:")
    for i, g in enumerate(groups): print(f"  Группа {i + 1}: k = {g} (размер {len(g)})")
    print("-" * 20)

    # 2. ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ
    max_abs_val = 0.0
    population_maps = []
    for i in range(num_modes_to_show):
        v_k = sorted_eigenvectors[:, i]
        rho_k = v_k.reshape((N, N), order='F')
        population_map = np.diag(rho_k).real.reshape((height, width))
        population_maps.append(population_map)

        # 2. КОРРЕКТНЫЙ РАСЧЕТ k: Находим V_max ТОЛЬКО среди НЕТРИВИАЛЬНЫХ мод (k>0)
    max_abs_val = 0.0
    # Убедимся, что есть хотя бы одна нетривиальная мода для анализа
    if num_modes_to_show > 1:
        # Срезаем первую (тривиальную) карту и ищем максимум среди остальных
        non_trivial_maps = np.array(population_maps[1:])
        if non_trivial_maps.size > 0:  # Доп. проверка, что массив не пустой
            max_abs_val = np.max(np.abs(non_trivial_maps))
    k_scaler = 1.0 / (max_abs_val + 1e-9)
    print(f"Глобальный масштабный коэффициент k_scaler = 1 / {max_abs_val:.4f} = {k_scaler:.4f}\n")

    # 3. ПОДГОТОВКА К РИСОВАНИЮ
    SCALING_DAMPENING_FACTOR = 0.4
    FIG_WIDTH = 20
    INFO_RATIO = 1.5
    DIAG_COL_RATIO = 2.0
    NUM_DIAG_COLUMNS = 4
    total_ratio = INFO_RATIO + DIAG_COL_RATIO * NUM_DIAG_COLUMNS
    diag_total_width = FIG_WIDTH * (DIAG_COL_RATIO * NUM_DIAG_COLUMNS) / total_ratio
    BASE_ROW_HEIGHT_FOR_4 = diag_total_width / NUM_DIAG_COLUMNS

    aspect_adjust = height / float(width)
    row_heights = [BASE_ROW_HEIGHT_FOR_4 * (NUM_DIAG_COLUMNS / len(g)) ** SCALING_DAMPENING_FACTOR * aspect_adjust for g in groups]
    total_fig_height = sum(row_heights)

    fig = plt.figure(figsize=(FIG_WIDTH, total_fig_height), dpi=150)
    main_gs = GridSpec(len(groups), 1, figure=fig, hspace=0.1, height_ratios=row_heights)

    # 4. ОСНОВНОЙ ЦИКЛ: Заполняем фигуру строками
    for i, group in enumerate(groups):
        num_diags = len(group)


        row_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=main_gs[i],
                                         width_ratios=[1.5, 2, 2, 2, 2], wspace=0.0)

        # Информационная ячейка
        ax_info = fig.add_subplot(row_gs[0])
        ax_info.axis('off')
        lambda_k_group = sorted_eigenvalues[group[0]]
        info_text = f"k' = {i + 1}\n\n    Re(λ) ≈\n {lambda_k_group.real:.4f}"
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=25)

        # создание осей (Axes) с объединением ячеек для каждой группы ***
        axes_to_draw_on = []
        if num_diags == 1:
            axes_to_draw_on.append(fig.add_subplot(row_gs[1:]))  # Объединяем все 4 ячейки
        elif num_diags == 2:
            axes_to_draw_on.append(fig.add_subplot(row_gs[1:3]))  # Объединяем 1-2
            axes_to_draw_on.append(fig.add_subplot(row_gs[3:]))  # Объединяем 3-4
        elif num_diags == 3:
            # Для 3-х диаграмм создаем свою под-сетку, чтобы они поделили место честно
            sub_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=row_gs[1:], wspace=0.0)
            axes_to_draw_on.append(fig.add_subplot(sub_gs[0]))
            axes_to_draw_on.append(fig.add_subplot(sub_gs[1]))
            axes_to_draw_on.append(fig.add_subplot(sub_gs[2]))
        elif num_diags == 4:
            # Для 4-х диаграмм используем каждую ячейку отдельно
            axes_to_draw_on.append(fig.add_subplot(row_gs[1]))
            axes_to_draw_on.append(fig.add_subplot(row_gs[2]))
            axes_to_draw_on.append(fig.add_subplot(row_gs[3]))
            axes_to_draw_on.append(fig.add_subplot(row_gs[4]))

        # Рисуем на созданных осях
        for j, k_original in enumerate(group):
            ax = axes_to_draw_on[j]
            lambda_k = sorted_eigenvalues[k_original]
            draw_single_mode_on_axis(
                ax=ax,
                population_map=population_maps[k_original],
                height=height, width=width,
                k_scaler=k_scaler, original_k=k_original,
                lambda_k=lambda_k
            )

    # 5. СОХРАНЕНИЕ И ОТОБРАЖЕНИЕ
    filename = "10x10_dynamically_grouped_diagrams_final_with_correct_norm_RESTORED2.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Все диаграммы сохранены в один файл: {filename}")

    plt.show()
    plt.close(fig)


# ==============================================================================
# --- БЛОК 3: ОСНОВНОЙ СКРИПТ ---
# ==============================================================================

def plot_color_bar_visualization(base_filename: str, label: str = "Relative amplitude ($p_k / max |p_k|$)"):
    """
    Создает цветовую шкалу, которая честно показывает, что цвет означает
    долю от максимального по модулю значения p_k.
    """
    num_points = 512
    values = np.linspace(-1.0, 1.0, num_points)

    # Строим строку пикселей, композитируя на белом фоне
    pixels = np.ones((1, num_points, 3), dtype=np.float64)  # белый фон
    for idx, v in enumerate(values):
        alpha = min(1.0, abs(v))
        if v > 1e-9:
            circle_rgb = np.array([1.0, 0.0, 0.0])  # красный
        elif v < -1e-9:
            circle_rgb = np.array([0.0, 0.0, 1.0])  # синий
        else:
            alpha = 0.0
            circle_rgb = np.array([1.0, 1.0, 1.0])
        pixels[0, idx, :] = alpha * circle_rgb + (1.0 - alpha) * np.array([1.0, 1.0, 1.0])

    fig, ax = plt.subplots(figsize=(16, 2), dpi=150)
    fig.patch.set_facecolor('white')

    ax.imshow(pixels, aspect='auto', extent=[-1, 1, 0, 1], origin='lower')
    ax.set_yticks([])

    # Оставляем числа от -1 до 1, но благодаря заголовку понятно, что это доли от максимума
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['-1.0', '-0.75', '-0.50', '-0.25', '0', '0.25', '0.50', '0.75', '1.0'],
                       fontsize=16, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_title(label, fontsize=22, pad=15)

    # Убираем лишние рамки
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='x', which='both', direction='out', length=5)

    base = base_filename
    if base.endswith('.png'):
        base = base[:-4]
    out_filename = base + "_bar_visualisation.png"

    plt.tight_layout()
    plt.savefig(out_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Цветовая шкала сохранена в: {out_filename}")
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # --- 1. Задаем параметры исследования (4x4, чтобы были интересные группы) ---
    HEIGHT = 10
    WIDTH = 10
    N = HEIGHT * WIDTH
    J = 1.0
    gamma = 0.1
    NUM_MODES_TO_VISUALIZE = 99

    # --- 2. Генерируем топологию и строим Лиувиллиан ---
    print(f"--- Исследование решетки {HEIGHT}x{WIDTH} (N={N}) ---")
    tau_grid = generate_grid_tau(HEIGHT, WIDTH)
    _, L_grid = build_liouvillian(tau_grid, J, gamma)
    print(tau_grid)
    # --- 3. Находим и сортируем моды ---
    print("Нахождение и сортировка мод...")
    sorted_lambdas, _, sorted_vs = analyze_liouvillian_modes(L_grid)
    print("Готово.")

    # --- 4. Вызываем новую функцию для создания изображения ---
    plot_dynamically_grouped_diagrams(
        sorted_eigenvalues=sorted_lambdas,
        sorted_eigenvectors=sorted_vs,
        height=HEIGHT,
        width=WIDTH,
        num_modes_to_show=NUM_MODES_TO_VISUALIZE
    )

    # --- 5. Создаем цветовую шкалу ---
    BAR_FILENAME = "10x10_dynamically_grouped_diagrams_final.png"
    plot_color_bar_visualization(BAR_FILENAME, label="relative population amplitude ($p_k / max|p_k|$)")


