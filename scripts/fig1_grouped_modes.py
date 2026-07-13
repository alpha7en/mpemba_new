"""Fig. 1 (big_modes_map): atlas of Liouvillian relaxation-mode population maps for a regular
10x10 lattice. Modes are grouped into degeneracy multiplets k' (same Re(lambda)) and laid out
in TWO super-columns (as in the published figure) to keep a near-page portrait aspect.

Each map is Re(diag(Mat(v_k))) rendered with the red(+)/white(0)/blue(-) convention, opacity =
|p_k|/max|p_k| (global normalization). Per-mode titles are dropped (kept only the group label
k'+Re(lambda)); a shared amplitude colorbar sits at the bottom.

The dense 10000x10000 eig is heavy, so the population maps are cached to precalc/ — the layout
can then be restyled instantly and the figure is reproducible. The computation is unchanged.
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import (
    analyze_liouvillian_modes_dense_strict,
    build_liouvillian_dense,
    draw_population_mode_on_axis,
    generate_grid_tau,
)
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

HEIGHT = WIDTH = 10
N = HEIGHT * WIDTH
J, GAMMA = 1.0, 0.1
NUM_MODES = 99

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig1_modes.npz"

# blue(-1) - white(0) - red(+1), matching the opacity-blended circle rendering
BWR = LinearSegmentedColormap.from_list("bwr_amp", [(0, "blue"), (0.5, "white"), (1.0, "red")])


def compute():
    """Dense eig of the regular-lattice Liouvillian -> population map of each mode. Unchanged."""
    tau = generate_grid_tau(HEIGHT, WIDTH)
    liouvillian = build_liouvillian_dense(tau, J, GAMMA)
    lambdas, _, right = analyze_liouvillian_modes_dense_strict(liouvillian)
    pop_maps = np.array([
        np.diag(right[:, k].reshape((N, N), order="F")).real.reshape(HEIGHT, WIDTH)
        for k in range(NUM_MODES)
    ])
    return lambdas[:NUM_MODES], pop_maps


def group_modes(lambdas):
    """Group non-trivial modes into multiplets sharing Re(lambda) (<=4 per group). Unchanged."""
    modes = list(range(1, NUM_MODES))
    groups, current = [], [modes[0]]
    for k in modes[1:]:
        if np.isclose(lambdas[k].real, lambdas[current[-1]].real) and len(current) < 4:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)
    return groups


def draw_group_row(fig, subspec, k_prime, lam_real, group, pop_maps, k_scaler):
    """One group: left label (k', Re(lambda)) + up to 4 mode maps spanning the 4 map slots."""
    row = GridSpecFromSubplotSpec(1, 5, subplot_spec=subspec, width_ratios=[1.8, 1, 1, 1, 1], wspace=0.04)
    ax_lbl = fig.add_subplot(row[0])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.5, f"$k'={k_prime}$\n$\\mathrm{{Re}}(\\lambda)$\n$\\approx {lam_real:.4f}$",
                ha="center", va="center", fontsize=8)

    m = len(group)
    if m == 1:
        axes = [fig.add_subplot(row[1:])]
    elif m == 2:
        axes = [fig.add_subplot(row[1:3]), fig.add_subplot(row[3:])]
    elif m == 3:
        sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=row[1:], wspace=0.04)
        axes = [fig.add_subplot(sub[i]) for i in range(3)]
    else:
        axes = [fig.add_subplot(row[1 + i]) for i in range(4)]

    for ax, k in zip(axes, group):
        draw_population_mode_on_axis(
            ax, pop_maps[k], HEIGHT, WIDTH, k_scaler,
            title_text="", radius=0.46, grid_linewidth=0.4,
            hide_spines=False, circle_edgecolor=None, circle_linewidth=0.0,
            title_fontsize=1, title_y=None,
        )


def main():
    apply_style()
    if CACHE.exists():
        z = np.load(CACHE)
        lambdas, pop_maps = z["lambdas"], z["pop_maps"]
    else:
        lambdas, pop_maps = compute()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, lambdas=lambdas, pop_maps=pop_maps)

    k_scaler = 1.0 / (np.max(np.abs(pop_maps[1:])) + 1e-9)
    groups = group_modes(lambdas)

    # split groups into two super-columns
    half = math.ceil(len(groups) / 2)
    columns = [(0, groups[:half]), (half, groups[half:])]

    w = 0.9 * WIDTH_FULL
    fig = plt.figure(figsize=(w, w * 1.42))
    outer = GridSpec(2, 2, height_ratios=[1, 0.025], hspace=0.06, wspace=0.06, figure=fig,
                     left=0.03, right=0.97, top=0.995, bottom=0.06)

    scaling_damp = 0.4
    for col_idx, (offset, col_groups) in enumerate(columns):
        rh = [(4.0 / len(g)) ** scaling_damp for g in col_groups]
        col_gs = GridSpecFromSubplotSpec(len(col_groups), 1, subplot_spec=outer[0, col_idx],
                                         height_ratios=rh, hspace=0.12)
        for r, group in enumerate(col_groups):
            draw_group_row(fig, col_gs[r], offset + r + 1, lambdas[group[0]].real, group, pop_maps, k_scaler)

    # shared amplitude colorbar
    cax = fig.add_subplot(outer[1, :])
    cb = fig.colorbar(ScalarMappable(Normalize(-1, 1), BWR), cax=cax, orientation="horizontal")
    cb.set_label(r"relative population amplitude ($p_k / \max|p_k|$)", fontsize=9)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / "10x10main_without_numbers.pdf"))


if __name__ == "__main__":
    main()
