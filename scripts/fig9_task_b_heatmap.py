"""Fig. 9/10 (Task B): pick a statistically representative rewired 10x10 graph (median L_avg
over an ensemble) and show (part1) the L_avg histogram used to select it and (part2) the local
excitability maps B(k,i) of the three slowest modes on that graph.

The heavy step (2000-graph search + dense Liouvillian eig) is cached to precalc/ so the figure
is reproducible and can be restyled without recomputing. The computation itself is unchanged.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map
from qdyn_research.network_metrics import calculate_average_shortest_path_length
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import generate_rewired_grid_tau_guaranteed_connectivity
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL, SEQ_CMAP

HEIGHT = WIDTH = 10
N = HEIGHT * WIDTH
J, GAMMA = 1.0, 0.1
TARGET_P = 0.15
NUM_SEARCH_ITERATIONS = 2000
MODES = [1, 2, 3]

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig9_representative.npz"
BASE = "Task_B_Strict_Layout_Fixed_test_p_0_ENG15_edited"


def compute():
    """2000-graph ensemble -> median-L_avg representative -> dense eig -> B(k,i). Unchanged."""
    candidates, metric_values = [], []
    for _ in range(NUM_SEARCH_ITERATIONS):
        tau = generate_rewired_grid_tau_guaranteed_connectivity(HEIGHT, WIDTH, TARGET_P)
        l_avg = calculate_average_shortest_path_length(tau)
        candidates.append(tau)
        metric_values.append(l_avg)
    metric_values = np.array(metric_values)
    best_idx = int(np.argmin(np.abs(metric_values - np.median(metric_values))))
    rep_tau = candidates[best_idx]

    liouvillian = build_liouvillian_dense(rep_tau, J, GAMMA)
    _, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)
    maps = np.stack([calculate_excitability_map(left_vecs, right_vecs, k, N) for k in MODES])
    return metric_values, rep_tau, float(metric_values[best_idx]), maps


def main():
    apply_style()

    if CACHE.exists():
        z = np.load(CACHE)
        metric_values, rep_tau, rep_val, maps = z["metric_values"], z["tau"], float(z["rep_val"]), z["maps"]
    else:
        metric_values, rep_tau, rep_val, maps = compute()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, metric_values=metric_values, tau=rep_tau, rep_val=rep_val, maps=maps)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # w = 0.75 * WIDTH_FULL
    w = WIDTH_FULL

    # --- part 1: L_avg selection histogram ---
    fig1, ax = plt.subplots(figsize=(w, w * 0.5), layout="constrained")
    ax.hist(metric_values, bins=25, color="0.7", edgecolor="0.35", linewidth=0.4)
    ax.axvline(rep_val, color="#D55E00", ls="--", lw=1.6, label="selected representative")
    ax.set_xlabel(r"$L_{avg}$")
    ax.set_ylabel("number of graphs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    save_pdf(fig1, str(FIG_DIR / f"{BASE}_part1.pdf"))

    # --- part 2: excitability maps B(k,i) for k=1,2,3 ---
    graph = nx.from_numpy_array(rep_tau)
    pos = {i: (i % WIDTH, (HEIGHT - 1) - (i // WIDTH)) for i in range(N)}
    fig2, axs = plt.subplots(1, 3, figsize=(w, w * 0.5), layout="constrained")
    for i, k in enumerate(MODES):
        # thin, de-emphasized edges on the dense 10x10 lattice: the accent is the node
        # excitability, so the graph structure should recede (not add visual noise).
        nx.draw_networkx_edges(graph, pos, ax=axs[i], edge_color="0.6", alpha=0.7, width=0.5)
        nc = nx.draw_networkx_nodes(graph, pos, ax=axs[i], node_color=maps[i], cmap=SEQ_CMAP,
                                    vmin=0, vmax=maps[i].max(), node_size=42, edgecolors="black", linewidths=0.3)
        axs[i].set_title(f"$k={k}$")
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        cb = fig2.colorbar(nc, ax=axs[i], orientation="horizontal", fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        if i == 1:
            cb.set_label(r"local excitability $B(k,i)$", fontsize=8)
    save_pdf(fig2, str(FIG_DIR / f"{BASE}_part2.pdf"))


if __name__ == "__main__":
    main()
