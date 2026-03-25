import networkx as nx
import numpy as np


# Classical graph metrics used by topology-focused figures.
def calculate_average_shortest_path_length(adj_matrix: np.ndarray) -> float:
    """Compute average shortest-path length on the largest connected component."""
    graph = nx.from_numpy_array(adj_matrix)
    if nx.is_connected(graph):
        return float(nx.average_shortest_path_length(graph))
    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)
    return float(nx.average_shortest_path_length(subgraph))


def run_average_path_experiment(height: int, width: int, n_runs: int, p_values: np.ndarray, generator):
    """Average path length over independent graph realizations for each rewiring probability p."""
    avg_lengths = {}
    for p in p_values:
        lengths = []
        for _ in range(n_runs):
            tau = generator(height, width, float(p))
            lengths.append(calculate_average_shortest_path_length(tau))
        avg_lengths[float(p)] = float(np.mean(lengths))
    return avg_lengths

