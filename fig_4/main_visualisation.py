"""

Тебе нужно решить следующую классическую задачу по теории сетей, используя python и его библиотеки.
У меня есть функция, генерирующая матрицу смежности (в моем случае это решетка 10х10 с некоторыми случайными модификациями, зависящями от входного параметра p, принимающего значения от 0 до 1)
Тебе нужно написать функцию, считающую среднюю длину минимального расстояния между узлами сети (классический параметр l теории сетей) убедись, что ты все верно рассчитал, либо сделай это полным перебором (сеть большая и можно себе это позволить, но лучше найди специализированную библиотеку для этого, что бы убедиться, что ты не допустил ошибок в этом важном научном коде).
Далее тебе нужно собрать данные следующим образом: ты генерируешь сеть при ФИКСИРОВАННОМ p N раз и считаешь ранее указанный параметр L, затем пот этим N значениям усредняешь L.
Таким образом проходимся по разным p от 0 до 1 и строим график <L> от p.    Функция, о которой шла речь:
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


"""


import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_grid_tau(height: int, width: int) -> np.ndarray:
    """Генерирует матрицу смежности для решетки height x width."""
    N = height * width
    tau = np.zeros((N, N), dtype=int)
    for i in range(height):
        for j in range(width):
            node_idx = i * width + j
            # Соединение с правым соседом
            if j < width - 1:
                right_idx = i * width + (j + 1)
                tau[node_idx, right_idx] = 1
                tau[right_idx, node_idx] = 1
            # Соединение с нижним соседом
            if i < height - 1:
                bottom_idx = (i + 1) * width + j
                tau[node_idx, bottom_idx] = 1
                tau[bottom_idx, node_idx] = 1
    return tau

def generate_rewired_grid_tau(height: int, width: int, p: float) -> np.ndarray:
    """
    Генерирует матрицу смежности для решетки 10x10 с некоторыми случайными
    модификациями, зависящими от входного параметра p.
    """
    initial_tau = generate_grid_tau(height, width)
    tau = initial_tau.copy()
    N = height * width
    edges = []
    [edges.append((i, j)) for i in range(N) for j in range(i + 1, N) if initial_tau[i, j] == 1]
    random.shuffle(edges)
    rewired_count = 0
    for u, v in edges:
        if random.random() < p:
            node_to_rewire = u if random.random() < 0.5 else v
            neighbors = np.where(tau[node_to_rewire, :] == 1)[0]
            forbidden_nodes = set(neighbors) | {node_to_rewire}
            valid_targets = [n for n in range(N) if n not in forbidden_nodes]
            if valid_targets:
                w = random.choice(valid_targets)
                # Удаляем старую связь
                tau[u, v] = tau[v, u] = 0
                # Добавляем новую связь
                tau[node_to_rewire, w] = tau[w, node_to_rewire] = 1
                rewired_count += 1
    # print(f"Перестроено {rewired_count} из {len(edges)} связей (p={p}).")
    return tau

def calculate_average_shortest_path_length(adj_matrix: np.ndarray) -> float:
    """
    Вычисляет среднюю длину кратчайшего пути для графа,
    заданного матрицей смежности.
    """
    G = nx.from_numpy_array(adj_matrix)
    # Проверяем, является ли граф связным. Если нет, то средняя длина
    # кратчайшего пути вычисляется для самой большой компоненты связности.
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        # Находим самую большую компоненту связности
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph)

def run_experiment(height: int, width: int, N_runs: int, p_values: np.ndarray) -> dict:
    """
    Проводит эксперимент по вычислению средней длины кратчайшего пути
    для разных значений p.
    """
    avg_lengths = {}
    for p in p_values:
        print(f"Обработка p = {p:.5f}")
        lengths_for_p = []
        for _ in range(N_runs):
            adj_matrix = generate_rewired_grid_tau(height, width, p)
            avg_length = calculate_average_shortest_path_length(adj_matrix)
            lengths_for_p.append(avg_length)
        avg_lengths[p] = np.mean(lengths_for_p)
    return avg_lengths

# --- Параметры эксперимента ---
HEIGHT = 10
WIDTH = 10
N_RUNS_PER_P = 1000  # Количество сетей для усреднения при каждом p

# --- ИЗМЕНЕНИЕ ЗДЕСЬ: Задаем логарифмическую шкалу для P ---
# Генерируем 20 точек от 10^-4 до 10^0 (что равно 1)
log_p_values = np.logspace(-4, 0, num=100)
# Добавляем 0 в начало массива значений p
P_VALUES = np.concatenate(([0.0], log_p_values))


# --- Запуск эксперимента ---
results = run_experiment(HEIGHT, WIDTH, N_RUNS_PER_P, P_VALUES)

# --- Визуализация результатов ---
p_vals = list(results.keys())
l_vals = list(results.values())

plt.figure(figsize=(10, 6))
plt.plot(p_vals, l_vals, marker='o', linestyle='-')
plt.xlabel('Вероятность перестройки (p)')
plt.ylabel('Средняя длина кратчайшего пути (<L>)')
plt.title('Зависимость средней длины кратчайшего пути от вероятности перестройки')
# --- ИЗМЕНЕНИЕ ЗДЕСЬ: Добавляем логарифмическую шкалу для оси X ---
plt.xscale('log')
plt.grid(True, which="both", ls="--") # Улучшаем сетку для лог. шкалы
plt.savefig(f"log{HEIGHT}x{WIDTH}_p_L.png")
plt.show()
