import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map
from qdyn_research.network_metrics import calculate_average_shortest_path_length
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import generate_rewired_grid_tau_guaranteed_connectivity


# ---------------------------
# System parameters
# ---------------------------
HEIGHT = 10
WIDTH = 10
N = HEIGHT * WIDTH
J = 1.0
GAMMA = 0.1

# ---------------------------
# Topological sampling parameters
# ---------------------------
TARGET_P = 0.15
NUM_SEARCH_ITERATIONS = 2000


def main():
    candidates = []
    metric_values = []

    # Representative network is selected by median L_avg over an ensemble at fixed p.
    for _ in range(NUM_SEARCH_ITERATIONS):
        tau = generate_rewired_grid_tau_guaranteed_connectivity(HEIGHT, WIDTH, TARGET_P)
        l_avg = calculate_average_shortest_path_length(tau)
        candidates.append({"tau": tau, "val": l_avg})
        metric_values.append(l_avg)

    target_val = np.median(metric_values)
    best_idx = np.argmin(np.abs(np.array(metric_values) - target_val))
    representative = candidates[best_idx]

    liouvillian = build_liouvillian_dense(representative["tau"], J, GAMMA)
    _, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)

    # B(k,i)=|<w_k|rho_i>/<w_k|v_k>|^2 for slow modes k=1,2,3.
    modes = [1, 2, 3]
    maps = {k: calculate_excitability_map(left_vecs, right_vecs, k, N) for k in modes}

    graph = nx.from_numpy_array(representative["tau"])
    pos = {i: (i % WIDTH, (HEIGHT - 1) - (i // WIDTH)) for i in range(N)}

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.5], hspace=0.3, wspace=0.3)

    scale_factor = (5.0 / float(WIDTH))
    node_size = 450 * (scale_factor ** 2)
    edge_width = 1.5 * scale_factor

    for i, k in enumerate(modes):
        ax = fig.add_subplot(gs[0, i])
        local_max = np.max(maps[k])

        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="gray", alpha=0.5, width=edge_width)
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=maps[k],
            cmap="inferno",
            vmin=0,
            vmax=local_max,
            node_size=node_size,
            edgecolors="black",
            linewidths=max(0.5, 1.5 * scale_factor),
        )

        ax.margins(0.1)
        ax.set_aspect("equal")
        ax.set_title(f"Mode k={k}\nB_max ~= {local_max:.2e}", fontsize=14)
        ax.axis("off")

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax,
            width="90%",
            height="5%",
            loc="lower center",
            bbox_to_anchor=(0, -0.08, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(nodes, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=10)
        if i == 1:
            cbar.set_label("Local excitability |c_k(rho_i)|^2", fontsize=12)

    ax_hist = fig.add_subplot(gs[1, :])
    ax_hist.hist(metric_values, bins=25, color="lightgreen", edgecolor="black", alpha=0.7, label="Graph ensemble")
    ax_hist.axvline(representative["val"], color="red", linestyle="--", linewidth=2, label="Selected representative")
    ax_hist.set_xlabel("Average shortest path length L_avg", fontsize=12)
    ax_hist.set_ylabel("Number of graphs", fontsize=12)
    ax_hist.set_title(f"Selection of representative graph (N={NUM_SEARCH_ITERATIONS})", fontsize=14)
    ax_hist.legend()
    ax_hist.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Excitability maps: representative case (p={TARGET_P})", fontsize=18, y=0.96)
    fig.savefig(f"Task_B_Strict_Layout_Fixed_test_p_0_ENG{int(TARGET_P * 100)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

