import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

# --- ИМПОРТЫ ---
# Предполагаем, что файлы лежат в одной директории или настроены пути
from reconecting.bright_dark.bright_dark_math import core_dense as core
from first_mode_analys import get_projection_coefficient, entropic_distance
from random_check import MpembaValidator

# --- КОНФИГУРАЦИЯ ---
HEIGHT = 6
WIDTH = 6
N = HEIGHT * WIDTH
J = 1.0
GAMMA = 0.1
TARGET_P = 0.15

# Параметры оптимизации (a=1, b=2 => Score = Gap^1 * Ratio^2)
POW_A = 1.0  # Степень для Gap (D_hot - D_cold)
POW_B = 2.0  # Степень для Ratio (c_cold / c_hot)

# Визуальный стиль
CMAP = 'inferno'
EDGE_COLOR = 'gray'
EDGE_ALPHA = 0.5
NODE_BORDER_COLOR = 'black'


def get_node_positions(height, width):
    """Координаты для решетки (инвертированный Y для соответствия матрице)"""
    pos = {}
    for i in range(height * width):
        row = i // width
        col = i % width
        pos[i] = (col, (height - 1) - row)
    return pos


def draw_base_graph(ax, G, pos, node_colors, scale_factor, title=None, vmax=None, title_color='black'):
    """Отрисовка графа с картой яркости"""
    calc_node_size = 450 * (scale_factor ** 2)
    calc_edge_width = 1.5 * scale_factor

    if vmax is None:
        vmax = np.max(node_colors)

    # Ребра
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR, alpha=EDGE_ALPHA,
        width=calc_edge_width
    )

    # Узлы
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        cmap=CMAP,
        vmin=0, vmax=vmax,
        node_size=calc_node_size,
        edgecolors=NODE_BORDER_COLOR,
        linewidths=max(0.5, 1.5 * scale_factor)
    )

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, color=title_color, fontweight=('bold' if title_color != 'black' else 'normal'))


def highlight_nodes(ax, G, pos, nodelist, color, scale_factor):
    """Обводка узлов"""
    calc_node_size = 450 * (scale_factor ** 2)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodelist,
        node_size=calc_node_size,
        node_color='none',
        edgecolors=color,
        linewidths=3.0 * scale_factor
    )


def add_stats_text(ax, lines, color='black', bg_color='white'):
    """Добавляет блок текста с параметрами внизу графика"""
    text_str = "\n".join(lines)
    ax.text(0.5, -0.02, text_str, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="gray", alpha=0.8))


def main():
    print(f"--- ЗАПУСК: Визуализация алгоритма (Score = Ratio^{POW_B} * Gap^{POW_A}) ---")

    # 1. ГЕНЕРАЦИЯ ДАННЫХ
    print("1. Генерация системы...")
    validator = MpembaValidator.load_state("res_gap\\pivoprosto.pkl")

    tau = validator.tau
    G = nx.from_numpy_array(tau)
    pos = get_node_positions(HEIGHT, WIDTH)
    scale_factor = (5.0 / float(WIDTH))
    L = validator.L
    evals, left_vs, right_vs = validator.evals, validator.left_vecs, validator.right_vecs
    # tau = core.generate_rewired_tau_dense(HEIGHT, WIDTH, TARGET_P)
    # G = nx.from_numpy_array(tau)
    # pos = get_node_positions(HEIGHT, WIDTH)
    # scale_factor = (5.0 / float(WIDTH))
#
    # L = core.build_liouvillian_dense(tau, J, GAMMA)
    # evals, left_vs, right_vs = core.analyze_modes_dense_strict(L)

    # Карты возбудимости
    maps = {}
    for k in [1, 2, 3]:
        maps[k] = core.calculate_excitability_map_dense(left_vs, right_vs, k, N)

    # Векторы для расчетов (Мода k=1)
    w_vec_1 = left_vs[:, 1]
    v_vec_1 = right_vs[:, 1]
    b_map_1 = maps[1]

    # --- ШАГ 1: ВИЗУАЛИЗАЦИЯ 3 МОД ---
    print("2. Рендеринг мод...")
    fig1 = plt.figure(figsize=(15, 5))
    gs1 = gridspec.GridSpec(1, 3, wspace=0.1)

    for i, k in enumerate([1, 2, 3]):
        ax = fig1.add_subplot(gs1[0, i])
        draw_base_graph(ax, G, pos, maps[k], scale_factor, title=f"Mode k={k}")

        # Colorbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="50%", height="5%", loc='lower center', borderpad=-1)
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, np.max(maps[k])), cmap=CMAP),
                     cax=axins, orientation='horizontal')
        axins.tick_params(labelsize=8)

    plt.suptitle("Step 1: Spectral analysis (Brightness maps)", fontsize=16)
    plt.savefig("Step1_Three_Modes.png", bbox_inches='tight', dpi=150)
    plt.close()

    # --- ШАГ 2: ГОРЯЧЕЕ СОСТОЯНИЕ (HOT) ---
    print("3. Выбор Hot...")
    hot_node = np.argmin(b_map_1)

    rho_hot_vec = np.zeros(N * N, dtype=complex)
    rho_hot_vec[hot_node * N + hot_node] = 1.0

    c1_hot = np.abs(get_projection_coefficient(w_vec_1, v_vec_1, rho_hot_vec))
    dist_hot = entropic_distance(rho_hot_vec, N)

    fig2, ax = plt.subplots(figsize=(6, 7))
    draw_base_graph(ax, G, pos, b_map_1, scale_factor, title="Step 2: Selection hot state (Dark Node)")
    highlight_nodes(ax, G, pos, [hot_node], 'red', scale_factor)

    stats = [
        f"Hot Node: {hot_node}",
        f"$|c_1| = {c1_hot:.2e}$",
        f"$D_{{hot}} = {dist_hot:.3f}$"
    ]
    add_stats_text(ax, stats, color='darkred')

    plt.savefig("Step2_Hot_State.png", bbox_inches='tight', dpi=150)
    plt.close()

    # --- ШАГ 3: ПОИСК ХОЛОДНОГО (COLD) ---
    print("4. Перебор M для Cold...")
    sorted_indices = np.argsort(b_map_1)[::-1]

    # Диапазон M
    M_values_to_show = list(range(2, min(11, N // 2 + 1)))
    n_plots = len(M_values_to_show)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig3 = plt.figure(figsize=(5 * cols, 6 * rows))
    gs3 = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.1)

    best_score = -1.0
    best_M = 0
    best_cold_info = {}

    for idx, M in enumerate(M_values_to_show):
        ax = fig3.add_subplot(gs3[idx // cols, idx % cols])

        # 1. Формируем смесь
        current_nodes = sorted_indices[:M]
        rho_mix_mat = np.zeros((N, N), dtype=complex)
        for node in current_nodes:
            rho_mix_mat[node, node] = 1.0 / M
        rho_mix_vec = rho_mix_mat.flatten()

        # 2. Считаем параметры
        c1_mix = np.abs(get_projection_coefficient(w_vec_1, v_vec_1, rho_mix_vec))
        dist_mix = entropic_distance(rho_mix_vec, N)

        # 3. Считаем Score
        # Условия валидности
        is_valid = (dist_mix < dist_hot)

        if is_valid:
            ratio = c1_mix / (c1_hot + 1e-15)
            gap = np.abs(dist_hot - dist_mix)
            # ФИЗИЧЕСКИ ОБОСНОВАННАЯ ФОРМУЛА
            score = (ratio ** POW_B) * (gap ** POW_A)
        else:
            ratio = 0
            gap = 0
            score = 0

        # 4. Проверка на лучший
        is_best = False
        if score > best_score:
            best_score = score
            best_M = M
            best_cold_info = {
                'nodes': current_nodes,
                'c1': c1_mix,
                'D': dist_mix,
                'ratio': ratio,
                'vec': rho_mix_vec
            }
            is_best = True
        is_best = False

        # 5. Отрисовка
        title_color = 'forestgreen' if is_best else 'black'
        title_text = f"M = {M} {'(BEST)' if is_best else ''}"

        draw_base_graph(ax, G, pos, b_map_1, scale_factor, title=title_text, title_color=title_color)
        highlight_nodes(ax, G, pos, current_nodes, 'blue', scale_factor)

        # Текстовая статистика
        stats_lines = [
            f"$|c_1| = {c1_mix:.3f}$",
            f"$D = {dist_mix:.3f}$",
            f"Gap $\\Delta D = {gap:.3f}$",
            f"Score = {score:.2f}"
        ]
        add_stats_text(ax, stats_lines, color='darkblue' if is_valid else 'gray')

    plt.suptitle(f"Step 3: Optimization $M$ (Score = Ratio$^{int(POW_B)}$ $\\times$ Gap$^{int(POW_A)}$)", fontsize=16)
    plt.savefig("Step3_Cold_Iterations.png", bbox_inches='tight', dpi=150)
    plt.close()

    # --- ШАГ 4: ФИНАЛ ---
    print(f"5. Final result (Best M={best_M})...")

    fig4, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.05, right=0.65, top=0.9, bottom=0.1)
    draw_base_graph(ax, G, pos, b_map_1, scale_factor, title="Algorithm result")

    # Рисуем оба состояния
    highlight_nodes(ax, G, pos, [hot_node], 'red', scale_factor)
    #highlight_nodes(ax, G, pos, best_cold_info['nodes'], 'blue', scale_factor)

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Hot (Dark, N=1)',
               markerfacecolor='none', markeredgecolor='red', markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label=f'Cold (Bright, N={best_M})',
               markerfacecolor='none', markeredgecolor='blue', markersize=12, markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)

    # Итоговая табличка
    final_stats = [
        "ПАРАМЕТРЫ ПАРЫ:",
        f"Hot:  $|c_1|={c1_hot:.2e}, D={dist_hot:.3f}$",
        f"Cold: $|c_1|={best_cold_info['c1']:.3f},  D={best_cold_info['D']:.3f}$",
        "----------------",
        f"Ratio ($|c_{{c}}|/|c_{{h}}|$): {best_cold_info['ratio']:.1f}",
        f"Gap ($D_h - D_c$): {(dist_hot - best_cold_info['D']):.3f}",
        f"FINAL SCORE: {best_score:.2f}"
    ]

    ax.text(1.05, 1.0, "\n".join(final_stats), transform=ax.transAxes,
            va='top', ha='left', fontsize=12, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    plt.savefig("Step4_Final_Algorithm_Selection.png", bbox_inches='tight', dpi=150)
    plt.close()

    print("Готово! Проверьте файлы PNG.")


if __name__ == "__main__":
    main()