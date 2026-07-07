import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from pathlib import Path

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map, entropic_distance, get_projection_coefficient
from qdyn_research.mpemba import find_guaranteed_mpemba_dense
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import generate_rewired_grid_tau_guaranteed_connectivity
from qdyn_research.plot_style import (apply_style, save_pdf, WIDTH_FULL, SEQ_CMAP,
                                      EDGE_WIDTH, EDGE_COLOR, EDGE_ALPHA)

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


# ---------------------------
# System parameters
# ---------------------------
HEIGHT = 6
WIDTH = 6
N = HEIGHT * WIDTH
J = 1.0
GAMMA = 0.1
TARGET_P = 0.15

# ---------------------------
# Optional cached-state file
# ---------------------------
CHECKPOINT = "res_gap/benchmark.pkl"

# ---------------------------
# Optimization exponents: Score = Ratio^POW_B * Gap^POW_A
# ---------------------------
POW_A = 1.0
POW_B = 2.0


def get_node_positions(height, width):
    return {i: (i % width, (height - 1) - (i // width)) for i in range(height * width)}


def draw_base_graph(ax, graph, pos, node_colors, scale_factor, title=None, vmax=None):
    node_size = 450 * (scale_factor ** 2)
    vmax = np.max(node_colors) if vmax is None else vmax

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=EDGE_COLOR, alpha=EDGE_ALPHA, width=EDGE_WIDTH)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        cmap=SEQ_CMAP,
        vmin=0,
        vmax=vmax,
        node_size=node_size,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def highlight_nodes(ax, graph, pos, nodes, color, scale_factor):
    node_size = 450 * (scale_factor ** 2)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=list(nodes),
        node_size=node_size,
        node_color="none",
        edgecolors=color,
        linewidths=3.0 * scale_factor,
    )


def load_or_build_system():
    if os.path.exists(CHECKPOINT):
        import joblib

        obj = joblib.load(CHECKPOINT)
        return obj.tau, obj.L, obj.evals, obj.left_vecs, obj.right_vecs

    tau = generate_rewired_grid_tau_guaranteed_connectivity(HEIGHT, WIDTH, TARGET_P)
    liouvillian = build_liouvillian_dense(tau, J, GAMMA)
    evals, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)
    return tau, liouvillian, evals, left_vecs, right_vecs


def main():
    apply_style()
    tau, _liouvillian, _evals, left_vecs, right_vecs = load_or_build_system()

    graph = nx.from_numpy_array(tau)
    pos = get_node_positions(HEIGHT, WIDTH)
    scale_factor = 5.0 / float(WIDTH)

    # Brightness map B(k,i)=|<w_k|rho_i>/<w_k|v_k>|^2 for the slow mode k=1.
    maps = {k: calculate_excitability_map(left_vecs, right_vecs, k, N) for k in [1, 2, 3]}
    b_map_1 = maps[1]

    fig1, axs1 = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.4), layout="constrained")
    for i, k in enumerate([1, 2, 3]):
        draw_base_graph(axs1[i], graph, pos, maps[k], scale_factor, title=f"$k={k}$")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig1, str(FIG_DIR / "Step1_Three_Modes.pdf"))

    vec_hot, vec_cold, vec_cold_score, history = find_guaranteed_mpemba_dense(
        left_vecs,
        right_vecs,
        N,
        excitability_map=b_map_1,
        mode_idx=1,
        distance_order="C",
        return_history=True,
    )

    hot_node = int(np.argmax(np.abs(vec_hot.reshape((N, N)).diagonal())))
    w_vec_1 = left_vecs[:, 1]
    v_vec_1 = right_vecs[:, 1]

    c_hot = np.abs(get_projection_coefficient(w_vec_1, v_vec_1, vec_hot))
    d_hot = entropic_distance(vec_hot, N, reshape_order="C")
    c_cold = np.abs(get_projection_coefficient(w_vec_1, v_vec_1, vec_cold_score))
    d_cold = entropic_distance(vec_cold_score, N, reshape_order="C")

    # Recover the cold-mixture node set (the M brightest nodes) behind the
    # score-optimal cold state, so Step 4 can label and highlight the final pair.
    valid_history = [h for h in history if h["valid"]]
    if valid_history:
        best_cold_entry = max(valid_history, key=lambda h: h["score"])
        cold_nodes = list(best_cold_entry["nodes"])
    else:
        cold_diag = np.abs(vec_cold_score.reshape((N, N)).diagonal())
        cold_nodes = list(np.where(cold_diag > 1e-9)[0])

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    draw_base_graph(ax2, graph, pos, b_map_1, scale_factor, title="Step 2: Hot node from dark region")
    highlight_nodes(ax2, graph, pos, [hot_node], "red", scale_factor)
    ax2.text(
        0.5,
        -0.04,
        f"Hot node: {hot_node} | |c1|={c_hot:.2e} | D_hot={d_hot:.3f}",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    fig2.savefig("Step2_Hot_State.png", bbox_inches="tight", dpi=150)
    plt.close(fig2)

    history_subset = [h for h in history if h["m"] <= 10]
    n_plots = len(history_subset)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig3 = plt.figure(figsize=(5 * cols, 6 * rows))
    gs3 = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.1)

    for idx, h in enumerate(history_subset):
        m = h["m"]
        current_nodes = h["nodes"]
        vec_mix = h["vec_mix"]
        c_mix = h["c_mix"]
        d_mix = h["d_mix"]
        score_m = h["score"]

        ax = fig3.add_subplot(gs3[idx // cols, idx % cols])
        draw_base_graph(ax, graph, pos, b_map_1, scale_factor, title=f"M = {m}")
        highlight_nodes(ax, graph, pos, current_nodes, "blue", scale_factor)
        ax.text(
            0.5,
            -0.04,
            f"|c1|={c_mix:.3f}  D={d_mix:.3f}  Score={score_m:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig3.suptitle("Step 3: Optimization over cold-mixture size M", fontsize=16)
    fig3.savefig("Step3_Cold_Iterations.png", bbox_inches="tight", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.05, right=0.65, top=0.9, bottom=0.1)
    draw_base_graph(ax4, graph, pos, b_map_1, scale_factor, title="Step 4: Final pair")
    highlight_nodes(ax4, graph, pos, [hot_node], "red", scale_factor)
    highlight_nodes(ax4, graph, pos, cold_nodes, "blue", scale_factor)

    best_cold_vec = vec_cold_score
    best_cold_c1 = c_cold  # projection coefficient for the fallback cold state
    best_cold_d = d_cold
    best_cold_ratio = best_cold_c1 / (c_hot + 1e-15)
    # Final panel follows the published workflow emphasis on the hot (dark) site.

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Hot (Dark, N=1)",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=12,
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Cold (Bright, N={len(cold_nodes)})",
            markerfacecolor="none",
            markeredgecolor="blue",
            markersize=12,
            markeredgewidth=2,
        ),
    ]
    ax4.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)

    ratio = best_cold_ratio
    gap = d_hot - best_cold_d
    final_score = (ratio ** POW_B) * (abs(gap) ** POW_A)
    final_stats = [
        "PAIR PARAMETERS:",
        f"Hot:  |c1|={c_hot:.2e}, D={d_hot:.3f}",
        f"Cold: |c1|={best_cold_c1:.3f}, D={best_cold_d:.3f}",
        "----------------",
        f"Ratio (|c_c|/|c_h|): {ratio:.2f}",
        f"Gap (D_h - D_c): {gap:.3f}",
        f"FINAL SCORE: {final_score:.3f}",
    ]
    ax4.text(
        1.05,
        1.0,
        "\n".join(final_stats),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9),
    )

    ax4.text(
        0.5,
        -0.14,
        f"Cold nodes={len(cold_nodes)} | |c1_cold|={best_cold_c1:.3f} | ratio={ratio:.2f} | Gap={gap:.3f}",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    fig4.savefig("Step4_Final_Algorithm_Selection.png", bbox_inches="tight", dpi=150)
    plt.close(fig4)


if __name__ == "__main__":
    main()

