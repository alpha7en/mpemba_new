import numpy as np
import matplotlib.pyplot as plt
import random


# --- ЯДРО: Функции для ПЛОТНЫХ матриц (на основе вашего файла) ---

def build_liouvillian_dense(tau: np.ndarray, J: complex, gamma: float) -> np.ndarray:
    """Строит ПЛОТНУЮ матрицу Лиувиллиана L с помощью numpy."""
    N = tau.shape[0]
    tau_dense = np.asarray(tau, dtype=np.complex128)
    H = -J / 2.0 * tau_dense
    I = np.eye(N, dtype=np.complex128)
    L_H = -1j * (np.kron(I, H) - np.kron(H.T, I))
    diag_LD_vec = np.full(N * N, -gamma, dtype=np.complex128)
    indices = np.arange(N)
    diag_LD_vec[indices * N + indices] = 0.0
    L_D = np.diag(diag_LD_vec)
    L = L_H + L_D
    return L


def analyze_liouvillian_modes_dense(L: np.ndarray):
    """
    Находит ВСЕ собственные значения плотного Лиувиллиана и сортирует их.
    """
    print("Используется плотный решатель np.linalg.eig. Это может занять много времени и памяти...")
    eigenvalues, eigenvectors = np.linalg.eig(L)
    sort_indices = np.argsort(eigenvalues.real)[::-1]
    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


# --- ЯДРО: Генераторы сетей (без изменений) ---

def generate_grid_tau(height: int, width: int) -> np.ndarray:
    N = height * width;
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        if (i + 1) % width != 0: tau[i, i + 1] = tau[i + 1, i] = 1
        if i < N - width: tau[i, i + width] = tau[i + width, i] = 1
    return tau


def generate_grid_with_manual_links_tau(height: int, width: int, extra_links: list[tuple[int, int]]) -> np.ndarray:
    tau = generate_grid_tau(height, width);
    N = height * width
    print(f"Добавление {len(extra_links)} ручных связей...")
    for u, v in extra_links:
        if 0 <= u < N and 0 <= v < N: tau[u, v] = tau[v, u] = 1
    return tau


def generate_rewired_grid_tau(height: int, width: int, p: float) -> np.ndarray:
    initial_tau = generate_grid_tau(height, width);
    tau = initial_tau.copy();
    N = height * width
    edges = [];
    [edges.append((i, j)) for i in range(N) for j in range(i + 1, N) if initial_tau[i, j] == 1]
    random.shuffle(edges);
    rewired_count = 0
    for u, v in edges:
        if random.random() < p:
            node_to_rewire = u if random.random() < 0.5 else v
            neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
            forbidden_nodes = set(neighbors) | {node_to_rewire}
            valid_targets = [n for n in range(N) if n not in forbidden_nodes]
            if valid_targets:
                w = random.choice(valid_targets)
                tau[u, v] = tau[v, u] = 0;
                tau[node_to_rewire, w] = tau[w, node_to_rewire] = 1;
                rewired_count += 1
    print(f"Перестроено {rewired_count} из {len(edges)} связей (p={p}).")
    return tau


# --- ЯДРО: Комбинированная визуализация (логика сохранена) ---

def plot_combined_results(final_tau: np.ndarray, eigenvalues: np.ndarray, height: int, width: int, mode: str, filename="",**kwargs):
    """Рисует комбинированный график: топология сети (сверху) и ее спектр (снизу)."""
    fig, (ax_topology, ax_spectrum) = plt.subplots(2, 1, figsize=(10, 20), gridspec_kw={'height_ratios': [1, 1]})
    plt.style.use('seaborn-v0_8-whitegrid')
    _draw_topology_on_ax(ax_topology, final_tau, height, width, mode, **kwargs)
    _draw_spectrum_on_ax(ax_spectrum, eigenvalues)
    plt.tight_layout(pad=4.0)

    if filename != "":
        plt.savefig(filename+".png")
    plt.show()


def _draw_topology_on_ax(ax, final_tau, height, width, mode, **kwargs):
    """Вспомогательная функция для отрисовки топологии на заданной оси."""
    N = height * width
    pos = {i: (i % width, - (i // width)) for i in range(N)}
    ax.set_title(f"Топология сети (Режим: {mode})", fontsize=16)
    ax.set_aspect('equal');
    ax.axis('off')
    # ... (логика отрисовки та же, что и в прошлом ответе)
    for i in range(N):
        for j in range(i + 1, N):
            if final_tau[i, j] == 1: ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color='lightgray',
                                             linewidth=1.5, zorder=0)
    highlighted_nodes, special_edges = set(), []
    if mode == 'manual_links':
        special_edges = kwargs.get('extra_links', [])
    elif mode == 'rewired':
        initial_tau = generate_grid_tau(height, width)
        for i in range(N):
            for j in range(i + 1, N):
                if final_tau[i, j] == 1 and initial_tau[i, j] == 0: special_edges.append((i, j))
        p_val = kwargs.get('p', 'N/A')
        ax.text(0.02, 0.98, f'p = {p_val}', transform=ax.transAxes, fontsize=14, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    if special_edges:
        colors = plt.cm.get_cmap('gist_rainbow', len(special_edges))
        for idx, (u, v) in enumerate(special_edges):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=colors(idx), linewidth=3, zorder=1)
            highlighted_nodes.update([u, v])
    node_colors = ['red' if i in highlighted_nodes else 'skyblue' for i in range(N)]
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=150, c=node_colors, edgecolors='black',
               zorder=2)


def _draw_spectrum_on_ax(ax, eigenvalues):
    """Вспомогательная функция для отрисовки спектра на заданной оси."""
    ax.scatter(eigenvalues.real, eigenvalues.imag, c='blue', alpha=0.7, edgecolors='k', s=50)
    ax.spines['left'].set_position('zero');
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none');
    ax.spines['top'].set_color('none')
    ax.set_xlabel("Re(λ)", loc='right');
    ax.set_ylabel("Im(λ)", loc='top', rotation=0)
    ax.set_title("Спектр собственных значений Лиувиллиана", fontsize=16);
    ax.grid(True)


# --- ОСНОВНОЙ БЛОК ИССЛЕДОВАНИЯ (адаптирован под плотные матрицы) ---
if __name__ == "__main__":
    # 1. Параметры симуляции
    # ВНИМАНИЕ: Используйте маленькие значения для плотных матриц!
    # 5x5 (N=25, L=625x625) - разумный предел для быстрого запуска.
    # 8x8 (N=64, L=4096x4096) - может занять несколько минут и потребовать > 5 ГБ ОЗУ.
    HEIGHT = 6
    WIDTH = 6
    N = HEIGHT * WIDTH
    J = 1.0
    gamma = 0.1
    # Для плотного решателя мы получаем ВСЕ значения, так что этот параметр не используется для вычислений
    NUM_MODES_TO_PLOT = 200

    MODE = 'rewired'

    print(f"--- Исследование решетки {HEIGHT}x{WIDTH} (N={N}) с ПЛОТНЫМИ МАТРИЦАМИ ---")

    # 2. Генерация матрицы смежности
    plot_kwargs = {}
    if MODE == 'grid':
        tau = generate_grid_tau(HEIGHT, WIDTH)
    elif MODE == 'manual_links':
        links_to_add = [(0, N - 1)]
        tau = generate_grid_with_manual_links_tau(HEIGHT, WIDTH, links_to_add)
        plot_kwargs['extra_links'] = links_to_add
    elif MODE == 'rewired':
        p_rewire = 0.03
        tau = generate_rewired_grid_tau(HEIGHT, WIDTH, p_rewire)
        plot_kwargs['p'] = p_rewire


    filename=f"{WIDTH}x{HEIGHT} {MODE}  spectrum p0_{int(p_rewire*10)}"
    # 3. Построение плотного Лиувиллиана
    print("Построение плотного Лиувиллиана...")
    L_dense = build_liouvillian_dense(tau, J, gamma)
    print(f"Готово. Размерность: {L_dense.shape}. Память: {L_dense.nbytes / 1024 ** 2:.2f} МБ")

    # 4. Нахождение мод
    sorted_lambdas, _ = analyze_liouvillian_modes_dense(L_dense)

    # 5. Комбинированная визуализация
    if sorted_lambdas is not None:
        print("Решатель завершил работу. Генерация комбинированного графика...")
        # Обрезаем количество точек на графике, если их слишком много для отрисовки
        lambdas_to_plot = sorted_lambdas[:NUM_MODES_TO_PLOT]
        plot_combined_results(tau, lambdas_to_plot, HEIGHT, WIDTH, MODE, **plot_kwargs, filename=filename)
        print("\n--- Исследование завершено. ---")