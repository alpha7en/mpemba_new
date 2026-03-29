import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from collections import defaultdict
import random
import networkx as nx
import community as community_louvain  # pip install python-louvain



def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Генерация базовой решетки."""
    N = height * width;
    tau = np.zeros((N, N), dtype=int)
    for i in range(N):
        if (i + 1) % width != 0: tau[i, i + 1] = tau[i + 1, i] = 1
        if i < N - width: tau[i, i + width] = tau[i + width, i] = 1
    return tau


def regenerate_rewired_tau(height: int, width: int, p: float, seed: int) -> np.ndarray:
    """
    Восстанавливает ТОЧНО ТУ ЖЕ сеть, которая была сгенерирована при сборе данных,
    используя тот же seed (run_idx). Это критично для поиска реальных сообществ.
    """
    random.seed(seed)

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
    return tau


def calculate_max_overlap(eigenvector: np.ndarray, communities: dict, N: int) -> float:
    """
    5.3. Максимальное перекрытие с сообществами (Max Overlap)
    Вычисляет максимальную долю "массы" моды, сосредоточенную в одном сообществе.
    """
    # 1. Восстанавливаем матрицу моды
    rho_k = eigenvector.reshape((N, N), order='F')

    # 2. Населенность (масса) в каждом узле: |(rho_k)_ii|^2
    population_distribution = np.abs(np.diag(rho_k)) ** 2

    # 3. Общая масса моды (знаменатель формулы)
    total_mass = np.sum(population_distribution)
    if np.isclose(total_mass, 0):
        return 0.0

    # 4. Считаем массу в каждом сообществе (числитель формулы)
    num_communities = max(communities.values()) + 1
    community_masses = np.zeros(num_communities)

    for node_idx, community_id in communities.items():
        community_masses[community_id] += population_distribution[node_idx]

    # 5. Overlap для каждого сообщества и поиск максимума
    overlaps = community_masses / total_mass
    return np.max(overlaps)


def find_latest_npz(pattern: str = "rewiring_spectrum_data_*.npz") -> str | None:
    """Find newest npz across common run locations."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cwd = Path.cwd()

    patterns = [
        str(cwd / pattern),
        str(script_dir / pattern),
        str(project_root / pattern),
        str(project_root / "**" / pattern),  # fallback recursive
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive="**" in p))

    # unique + existing only
    files = [str(Path(f).resolve()) for f in files if Path(f).is_file()]
    files = list(dict.fromkeys(files))
    if not files:
        print("ERROR: .npz files not found.")
        print(f"cwd        : {cwd}")
        print(f"script_dir : {script_dir}")
        print("checked patterns:")
        for p in patterns:
            print(f"  - {p}")
        return None

    return max(files, key=os.path.getctime)

# --- ============================================= ---
# --- ОСНОВНОЙ БЛОК: Загрузка, обработка, визуализация ---
# --- ============================================= ---
if __name__ == "__main__":
    # 1. ПОИСК ФАЙЛА С ДАННЫМИ
    latest_file = find_latest_npz()
    if latest_file is None:
        raise SystemExit(1)

    print(f"Loading data from: '{latest_file}'\n")
    data = np.load(latest_file)

    # 2. РАЗБОР И ГРУППИРОВКА ДАННЫХ
    grouped_data = defaultdict(list)
    p_values_from_file = set()

    for key in data.keys():
        if key.endswith('_vectors'):
            base_key = key.replace('_vectors', '')
            parts = base_key.split('_')
            p_val = float(parts[1])
            run_idx = int(parts[3])

            p_values_from_file.add(p_val)
            grouped_data[p_val].append({'run_idx': run_idx, 'vectors': data[key]})

    p_values = sorted(list(p_values_from_file))

    # Определяем параметры N, HEIGHT, WIDTH
    first_vector_set = next(iter(grouped_data.values()))[0]['vectors']
    N_squared, num_modes_k = first_vector_set.shape
    N = int(np.sqrt(N_squared))
    HEIGHT = WIDTH = int(np.sqrt(N))  # Предполагаем квадратную решетку

    print(f"Параметры:\n  - N={N} ({HEIGHT}x{WIDTH})\n  - k={num_modes_k}\n  - p={p_values}\n")

    # 3. ВЫЧИСЛЕНИЕ MaxOverlap
    modes_to_analyze = [1, 2, 3]  # Первые три нетривиальные моды
    overlap_results = defaultdict(dict)

    for mode_idx in modes_to_analyze:
        print(f"Анализ моды λ_{mode_idx}...")
        avg_overlaps, std_overlaps = [], []

        for p in p_values:
            max_overlaps_for_p = []
            for run_data in grouped_data[p]:
                run_idx = run_data['run_idx']

                # А. Восстанавливаем топологию сети для данного запуска
                tau = regenerate_rewired_tau(HEIGHT, WIDTH, p, seed=run_idx)
                G = nx.from_numpy_array(tau)

                # Б. Находим сообщества алгоритмом Лувена
                communities = community_louvain.best_partition(G)

                # В. Вычисляем MaxOverlap для вектора из файла
                eigenvector = run_data['vectors'][:, mode_idx]
                max_overlap = calculate_max_overlap(eigenvector, communities, N)
                max_overlaps_for_p.append(max_overlap)

            # Усредняем по запускам
            avg_overlaps.append(np.mean(max_overlaps_for_p))
            std_overlaps.append(np.std(max_overlaps_for_p))

        overlap_results[mode_idx]['avg'] = avg_overlaps
        overlap_results[mode_idx]['std'] = std_overlaps

    # 4. ПОСТРОЕНИЕ ГРАФИКОВ
    print("\n--- Генерация графиков MaxOverlap от p ---")
    for mode_idx in modes_to_analyze:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.errorbar(
            p_values,
            overlap_results[mode_idx]['avg'],
            yerr=overlap_results[mode_idx]['std'],
            fmt='-o', capsize=5,
            label=f'Средний MaxOverlap',
            color='darkred' if mode_idx == 1 else 'darkgreen' if mode_idx == 2 else 'darkblue'
        )

        ax.set_title(f"Локализация моды λ_{mode_idx} на сообществах", fontsize=16)
        ax.set_xlabel("Вероятность перестроения (p)", fontsize=12)
        ax.set_ylabel(f"⟨MaxOverlap(λ_{mode_idx})⟩", fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        ax.set_xscale('log')
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout()
        plt.savefig(f"MaxOverlap_mode_k{mode_idx}_{HEIGHT}x{WIDTH}.png")
        plt.show()