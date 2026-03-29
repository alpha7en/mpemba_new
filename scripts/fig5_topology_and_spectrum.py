from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.spectral import analyze_liouvillian_modes_dense
from qdyn_research.topology import generate_grid_tau, generate_grid_with_manual_links_tau, generate_rewired_grid_tau


def draw_topology(ax, tau, height, width, mode, extra_links=None, p=None):
    n = height * width
    pos = {i: (i % width, -(i // width)) for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if tau[i, j] == 1:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="lightgray", linewidth=1.5, zorder=0)

    highlighted_nodes = set()
    special_edges = []

    if mode == "manual_links" and extra_links is not None:
        special_edges = extra_links
    elif mode == "rewired":
        initial_tau = generate_grid_tau(height, width)
        for i in range(n):
            for j in range(i + 1, n):
                if tau[i, j] == 1 and initial_tau[i, j] == 0:
                    special_edges.append((i, j))

    if special_edges:
        colors = plt.get_cmap("gist_rainbow", len(special_edges))
        for idx, (u, v) in enumerate(special_edges):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=colors(idx), linewidth=3, zorder=1)
            highlighted_nodes.update([u, v])

    node_colors = ["red" if i in highlighted_nodes else "skyblue" for i in range(n)]
    ax.scatter([p_[0] for p_ in pos.values()], [p_[1] for p_ in pos.values()], s=150, c=node_colors, edgecolors="black", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    if mode != "rewired":
        ax.set_title(f"Топология сети (Режим: {mode})", fontsize=16, fontweight="bold")


def draw_spectrum(ax, eigenvalues):
    ax.scatter(eigenvalues.real, eigenvalues.imag, c="blue", alpha=0.7, edgecolors="k", s=50)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlabel(r"$\mathbf{Re}\,\boldsymbol{\lambda}$", loc="right", fontsize=20, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{Im}\,\boldsymbol{\lambda}$", fontsize=20, fontweight="bold")
    ax.set_title("Eigenvalue spectrum of Liouvillian", fontsize=25, fontweight="bold")

    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=20, length=4, width=1.2, direction="inout")
    ax.tick_params(axis="both", which="minor", length=2, width=1.0, direction="inout")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # hide duplicate 0 at origin on Y axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: "" if abs(val) < 1e-12 else f"{val:g}"))

    # center X coordinate of Y-label on the moved vertical axis (x=0)
    x0_display = ax.transData.transform((0, 0))[0]
    x0_axes = ax.transAxes.inverted().transform((x0_display, 0))[0]

    # move Y-label upward to approximately y=3.7 (in data coordinates)
    y_target_display = ax.transData.transform((0, 3.7))[1]
    y_target_axes = ax.transAxes.inverted().transform((0, y_target_display))[1]

    # shift label a bit to the right of the vertical axis to avoid overlap with y tick labels
    x_shift_axes = 0.05
    ax.yaxis.set_label_coords(x0_axes + x_shift_axes, y_target_axes)
    ax.yaxis.label.set_horizontalalignment("left")

    ax.grid(False)


def main():
    # ---------------------------
    # System and plotting parameters
    # ---------------------------
    height =10
    width = 10
    j = 1.0
    gamma = 0.1
    num_modes_to_plot = 200
    mode = "rewired"

    if mode == "grid":
        tau = generate_grid_tau(height, width)
        kwargs = {}
        filename = f"{width}x{height} {mode} spectrum"
    elif mode == "manual_links":
        links = [(0, height * width - 1)]
        tau = generate_grid_with_manual_links_tau(height, width, links)
        kwargs = {"extra_links": links}
        filename = f"{width}x{height} {mode} spectrum"
    else:
        p_rewire = 0.1
        tau = generate_rewired_grid_tau(height, width, p_rewire)
        kwargs = {"p": p_rewire}
        filename = f"{width}x{height} {mode}  spectrum p0_{int(p_rewire * 100)}"

    # Liouvillian spectrum lambda_k is plotted in the complex plane (Re, Im).
    liouvillian = build_liouvillian_dense(tau, j, gamma)
    lambdas, _ = analyze_liouvillian_modes_dense(liouvillian)
    lambdas_to_plot = lambdas[:num_modes_to_plot]

    fig, (ax_topo, ax_spec) = plt.subplots(2, 1, figsize=(10, 20), gridspec_kw={"height_ratios": [1, 1]})
    plt.style.use("seaborn-v0_8-whitegrid")
    draw_topology(ax_topo, tau, height, width, mode, **kwargs)
    draw_spectrum(ax_spec, lambdas_to_plot)
    fig.tight_layout(pad=4.0)
    fig.savefig(f"{filename}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

