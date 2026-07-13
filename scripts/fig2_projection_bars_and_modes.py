"""Fig. 2 (9states -> 8 states): modal content of the candidate initial states on a 10x10
lattice. Each panel = a schematic of the initial photon distribution + a log-scale bar chart
of the mode-contribution weights W_k = |c_k|^2 (eq. c_k). The spatial mode maps of the old
version are dropped (not discussed in the text, redundant with Fig.1). Only the plotting is
new; the projection computation is unchanged and cached (dense 10x10 eig is expensive).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import (
    analyze_liouvillian_modes_dense_strict,
    build_liouvillian_dense,
    calculate_distance_metric_logm,
    project_rho_on_modes,
    generate_grid_tau,
    create_localized_state,
    create_opposite_corners_state,
    create_four_corners_state,
    create_mixed_diagonal_state,
    create_entangled_diagonal_state,
    create_inner_corners_state,
    create_top_bottom_edges_state,
    create_checkerboard_state,
)
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig2_projections.npz"
N = 10
J, GAMMA = 1.0, 0.1

# 8 states (the article caption says "eight"; the legacy "Boundary" state is dropped).
STATES = [
    ("Center", create_localized_state),
    ("Opp. corners", create_opposite_corners_state),
    ("Four corners", create_four_corners_state),
    ("Mixed diag.", create_mixed_diagonal_state),
    ("Coherent diag.", create_entangled_diagonal_state),
    ("Inner corners", create_inner_corners_state),
    ("Edges (t/b)", create_top_bottom_edges_state),
    ("Checkerboard", create_checkerboard_state),
]


def analyze_projections(coefficients):
    """Normalized W_k = |c_k|^2 over the non-trivial modes, plus the max index of the top-8."""
    contributions = np.abs(coefficients[1:]) ** 2
    total = np.sum(contributions)
    if total < 1e-12:
        return np.zeros_like(contributions), 8
    normalized = contributions / total
    mode_indices = np.arange(1, len(normalized) + 1)
    top_8 = mode_indices[np.argsort(normalized)[::-1]][:8]
    return normalized, int(np.max(top_8))


def compute():
    """Dense 10x10 eig -> modal projection of each state (the expensive, cached part)."""
    tau = generate_grid_tau(N, N)
    liouvillian = build_liouvillian_dense(tau, J, GAMMA)
    _, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)
    contribs, kmax = [], []
    for _name, gen in STATES:
        rho, _idx = gen(N, N)
        coeffs = project_rho_on_modes(rho, left_vecs, right_vecs, order="F")
        norm, km = analyze_projections(coeffs)
        contribs.append(norm)
        kmax.append(km)
    return np.array(contribs), np.array(kmax)


def draw_bar(ax, contribs, k_max, y_top, show_xlabel, show_ylabel):
    ax.set_yscale("log")
    ymin = 0.01
    modes = np.arange(1, k_max + 1)
    pct = contribs[:k_max] * 100.0
    keep = pct > ymin
    if k_max <= 40:
        ax.bar(modes[keep], pct[keep], color="#0072B2", width=0.85)
    else:
        ax.vlines(modes[keep], ymin, pct[keep], color="#0072B2", linewidth=0.7)
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_ylim(ymin, y_top)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", which="major", ls="--", lw=0.4, alpha=0.6)
    if show_xlabel:
        ax.set_xlabel("mode $k$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$W_k$ (%)", fontsize=8)


def main():
    apply_style()

    if CACHE.exists():
        z = np.load(CACHE)
        contribs_all, kmax_all = z["contribs"], z["kmax"]
    else:
        contribs_all, kmax_all = compute()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, contribs=contribs_all, kmax=kmax_all)

    # cheap per-state extras (schematic indices + distance metric), then sort by D
    panels = []
    for i, (name, gen) in enumerate(STATES):
        rho, idx = gen(N, N)
        panels.append({
            "name": name,
            "idx": idx,
            "D": calculate_distance_metric_logm(rho),
            "contribs": contribs_all[i],
            "kmax": int(kmax_all[i]),
        })
    panels.sort(key=lambda p: p["D"], reverse=True)

    y_top = max(p["contribs"].max() for p in panels) * 100 * 1.6

    fig = plt.figure(figsize=(WIDTH_FULL, WIDTH_FULL * 1.2))
    outer = gridspec.GridSpec(4, 2, figure=fig, left=0.10, right=0.985, top=0.96,
                              bottom=0.085, hspace=0.7, wspace=0.30)

    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        # bar chart on the LEFT (its y-axis has room); schematic on the RIGHT (no collisions)
        cell = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[r, c],
                                                width_ratios=[3.0, 1], wspace=0.15)
        ax_b = fig.add_subplot(cell[0])
        draw_bar(ax_b, p["contribs"], p["kmax"], y_top,
                 show_xlabel=(r == 3), show_ylabel=(c == 0))
        ax_b.set_title(f"{p['name']}  ($D_0$={p['D']:.2f})", fontsize=8, pad=3)

        ax_s = fig.add_subplot(cell[1])
        sch = np.zeros((N, N))
        for j in p["idx"]:
            sch[divmod(j, N)] = 1.0
        ax_s.imshow(sch, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax_s.set_xticks([]); ax_s.set_yticks([])

    save_pdf(fig, str(FIG_DIR / "new_tabled_10x10_bar_chart_log_ENG_new_8STATES.pdf"))


if __name__ == "__main__":
    main()
