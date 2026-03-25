from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt

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
        ax.text(0.02, 0.98, f"p = {p}", transform=ax.transAxes, fontsize=14, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    if special_edges:
        colors = plt.get_cmap("gist_rainbow", len(special_edges))
        for idx, (u, v) in enumerate(special_edges):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=colors(idx), linewidth=3, zorder=1)
            highlighted_nodes.update([u, v])

    node_colors = ["red" if i in highlighted_nodes else "skyblue" for i in range(n)]
    ax.scatter([p_[0] for p_ in pos.values()], [p_[1] for p_ in pos.values()], s=150, c=node_colors, edgecolors="black", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Топология сети (Режим: {mode})", fontsize=16)


def draw_spectrum(ax, eigenvalues):
    ax.scatter(eigenvalues.real, eigenvalues.imag, c="blue", alpha=0.7, edgecolors="k", s=50)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xlabel("Re(lambda)", loc="right")
    ax.set_ylabel("Im(lambda)", loc="top", rotation=0)
    ax.set_title("Спектр собственных значений Лиувиллиана", fontsize=16)
    ax.grid(True)


def main():
    # ---------------------------
    # System and plotting parameters
    # ---------------------------
    height = 6
    width = 6
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
        p_rewire = 0.03
        tau = generate_rewired_grid_tau(height, width, p_rewire)
        kwargs = {"p": p_rewire}
        filename = f"{width}x{height} {mode}  spectrum p0_{int(p_rewire * 10)}"

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

