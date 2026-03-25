import numpy as np

from .metrics import entropic_distance, get_projection_coefficient


# Mpemba search helpers used by fig_9_10_11 scripts.
def find_guaranteed_mpemba_dense(left_vecs, right_vecs, n_sites, excitability_map, mode_idx=1, distance_order="C"):
    """Select hot/cold pair by maximizing projection ratio and score over bright-node mixtures."""
    w_vec = left_vecs[:, mode_idx]
    v_vec = right_vecs[:, mode_idx]

    dark_node = int(np.argmin(excitability_map))
    rho_hot = np.zeros((n_sites, n_sites), dtype=complex)
    rho_hot[dark_node, dark_node] = 1.0
    vec_hot = rho_hot.flatten(order=distance_order)

    c_hot = np.abs(get_projection_coefficient(w_vec, v_vec, vec_hot))
    d_hot = entropic_distance(vec_hot, n_sites, reshape_order=distance_order)

    sorted_nodes = np.argsort(excitability_map)[::-1]

    best_ratio = -1.0
    best_score = -1.0
    best_ratio_vec = None
    best_score_vec = None

    for m in range(2, max(3, int(n_sites / 2))):
        chosen = sorted_nodes[:m]
        rho_cold = np.zeros((n_sites, n_sites), dtype=complex)
        for node in chosen:
            rho_cold[node, node] = 1.0 / m
        vec_cold = rho_cold.flatten(order=distance_order)

        c_cold = np.abs(get_projection_coefficient(w_vec, v_vec, vec_cold))
        d_cold = entropic_distance(vec_cold, n_sites, reshape_order=distance_order)

        if d_cold >= d_hot - 0.01:
            continue

        ratio = (c_cold / (c_hot + 1e-15)) ** 2
        score = ratio * np.abs(d_cold - d_hot)

        if ratio > best_ratio:
            best_ratio = ratio
            best_ratio_vec = vec_cold
        if score > best_score:
            best_score = score
            best_score_vec = vec_cold

    if best_ratio_vec is None:
        rho_def = np.zeros((n_sites, n_sites), dtype=complex)
        rho_def[sorted_nodes[0], sorted_nodes[0]] = 1.0
        best_ratio_vec = rho_def.flatten(order=distance_order)

    if best_score_vec is None:
        best_score_vec = best_ratio_vec

    return vec_hot, best_ratio_vec, best_score_vec

