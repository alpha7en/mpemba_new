import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from reconecting.bright_dark.bright_dark_math import core_dense as core

# --- ПАРАМЕТРЫ ---
HEIGHT = 10
WIDTH = 10
N = HEIGHT * WIDTH
J = 1.0
GAMMA = 0.1

TARGET_P = 0.15
NUM_SEARCH_ITERATIONS = 2000


def calculate_average_shortest_path_length(adj_matrix: np.ndarray) -> float:
    G = nx.from_numpy_array(adj_matrix)
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph)


def main():
    print(f"ЗАПУСК ЗАДАЧИ Б (Strict Layout). p={TARGET_P}")

    candidates = []
    metric_values = []

    # 1. Топологический отбор
    print(f"Генерация {NUM_SEARCH_ITERATIONS} графов...")
    for i in range(NUM_SEARCH_ITERATIONS):
        tau = core.generate_rewired_tau_dense(HEIGHT, WIDTH, TARGET_P)
        avg_path = calculate_average_shortest_path_length(tau)
        candidates.append({'tau': tau, 'val': avg_path})
        metric_values.append(avg_path)

    # Медианный отбор
    target_val = np.median(metric_values)
    best_idx = np.argmin(np.abs(np.array(metric_values) - target_val))
    rep = candidates[best_idx]

    print(f"Выбран граф #{best_idx}. L_avg={rep['val']:.4f}")

    # 2. Квантовый расчет (Core verified)
    L = core.build_liouvillian_dense(rep['tau'], J, GAMMA)
    _, left_vs, right_vs = core.analyze_modes_dense_strict(L)

    modes_to_plot = [1, 2, 3]
    maps = {}
    for k in modes_to_plot:
        # Используется проверенная формула из dense_verified_core
        maps[k] = core.calculate_excitability_map_dense(left_vs, right_vs, k, N)

    # 3. Визуализация (Strict Grid Layout)
    G = nx.from_numpy_array(rep['tau'])

    # Координаты для решетки
    pos = {}
    for i in range(N):
        row = i // WIDTH
        col = i % WIDTH
        pos[i] = (col, (HEIGHT - 1) - row)  # Y инвертирован для соответствия матрице

    # Настройка GridSpec:
    # 3 строки:
    # 0: Графики (вес 1)
    # 1: Colorbars (вес 0.05 - узкие)
    # 2: Гистограмма (вес 0.6)
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5], hspace=0.3, wspace=0.3)

    scale_factor = (5.0 / float(WIDTH))

    # Размер узла: для 5x5 = 450, для 10x10 = 450 * (1/2)^2 = 112.5
    calc_node_size = 450 * (scale_factor ** 2)

    # Толщина линии: линейное уменьшение
    calc_edge_width = 1.5 * scale_factor

    for i, k in enumerate(modes_to_plot):
        ax = fig.add_subplot(gs[0, i])

        local_max = np.max(maps[k])

        # Рисуем связи (тоньше для больших сеток)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray', alpha=0.5,
            width=calc_edge_width
        )

        # Рисуем узлы (меньше для больших сеток)
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=maps[k],
            cmap='inferno',
            vmin=0, vmax=local_max,
            node_size=calc_node_size,
            edgecolors='black',
            linewidths=max(0.5, 1.5 * scale_factor)  # Обводка тоже тоньше
        )

        # Добавляем отступы, чтобы узлы не касались краев графика
        ax.margins(0.1)

        # Гарантия квадратности
        ax.set_aspect('equal')
        ax.set_title(f"Mode $k={k}$\n$B_{{\\max}} \\approx {local_max:.2e}$", fontsize=14)
        ax.axis('off')

        # Colorbar через inset_axes (как в прошлом исправлении)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax,
                           width="90%",
                           height="5%",
                           loc='lower center',
                           bbox_to_anchor=(0, -0.08, 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0)

        cbar = fig.colorbar(nodes, cax=axins, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)

        if i == 1:
            cbar.set_label(r"Local excitability $|c_k(\rho_i)|^2$", fontsize=12)

    # --- Ряд 3: Гистограмма (на всю ширину) ---
    ax_hist = fig.add_subplot(gs[1, :])
    ax_hist.hist(metric_values, bins=25, color='lightgreen', edgecolor='black', alpha=0.7, label='Graph ensemble')
    ax_hist.axvline(rep['val'], color='red', linestyle='--', linewidth=2, label='Selected representative')

    ax_hist.set_xlabel(r"Average shortest path length $L_{avg}$", fontsize=12)
    ax_hist.set_ylabel("Number of graphs", fontsize=12)
    ax_hist.set_title(f"Selection of a representative graph based on topology ($N={NUM_SEARCH_ITERATIONS}$)", fontsize=14)
    ax_hist.legend()
    ax_hist.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Excitability maps: Representative case ($p={TARGET_P}$)", fontsize=18, y=0.96)

    plt.savefig(f'Task_B_Strict_Layout_Fixed_test_p_0_ENG{int(TARGET_P*100)}.png', dpi=150, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    main()