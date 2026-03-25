from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from qdyn_research import (
    analyze_liouvillian_modes_dense_strict,
    build_liouvillian_dense,
    draw_population_mode_on_axis,
    generate_grid_tau,
    generate_relative_amplitude_colorbar,
)


# Figure 1: grouped population maps for slow Liouvillian modes.
def plot_dynamically_grouped_diagrams(sorted_eigenvalues, sorted_eigenvectors, height, width, num_modes_to_show):
    n = height * width
    modes = list(range(1, min(num_modes_to_show, n * n)))
    if not modes:
        return

    # Group modes by near-degenerate Re(lambda), up to four modes per group.
    groups = []
    current_group = [modes[0]]
    for idx in range(1, len(modes)):
        k_curr = modes[idx]
        k_prev = current_group[-1]
        if np.isclose(sorted_eigenvalues[k_curr].real, sorted_eigenvalues[k_prev].real) and len(current_group) < 4:
            current_group.append(k_curr)
        else:
            groups.append(current_group)
            current_group = [k_curr]
    groups.append(current_group)

    # Population extraction uses column-major reshape to stay consistent with vec(rho).
    population_maps = []
    for k in range(num_modes_to_show):
        rho_k = sorted_eigenvectors[:, k].reshape((n, n), order="F")
        population_maps.append(np.diag(rho_k).real.reshape((height, width)))

    max_abs = 0.0
    if num_modes_to_show > 1:
        non_trivial = np.array(population_maps[1:])
        if non_trivial.size > 0:
            max_abs = np.max(np.abs(non_trivial))
    k_scaler = 1.0 / (max_abs + 1e-9)

    scaling_damp = 0.4
    fig_width = 20
    info_ratio = 1.5
    diag_ratio = 2.0
    num_diag_cols = 4
    total_ratio = info_ratio + diag_ratio * num_diag_cols
    diag_total_width = fig_width * (diag_ratio * num_diag_cols) / total_ratio
    base_row_height = diag_total_width / num_diag_cols
    aspect_adjust = height / float(width)
    row_heights = [base_row_height * (num_diag_cols / len(g)) ** scaling_damp * aspect_adjust for g in groups]

    fig = plt.figure(figsize=(fig_width, sum(row_heights)), dpi=150)
    main_gs = GridSpec(len(groups), 1, figure=fig, hspace=0.1, height_ratios=row_heights)

    for row_idx, group in enumerate(groups):
        row_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=main_gs[row_idx], width_ratios=[1.5, 2, 2, 2, 2], wspace=0.0)
        ax_info = fig.add_subplot(row_gs[0])
        ax_info.axis("off")
        lam = sorted_eigenvalues[group[0]]
        ax_info.text(0.5, 0.5, f"k' = {row_idx + 1}\n\n    Re(λ) ≈\n {lam.real:.4f}", ha="center", va="center", fontsize=25)

        axes = []
        if len(group) == 1:
            axes.append(fig.add_subplot(row_gs[1:]))
        elif len(group) == 2:
            axes.append(fig.add_subplot(row_gs[1:3]))
            axes.append(fig.add_subplot(row_gs[3:]))
        elif len(group) == 3:
            sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=row_gs[1:], wspace=0.0)
            axes.extend([fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[2])])
        else:
            axes.extend([fig.add_subplot(row_gs[1]), fig.add_subplot(row_gs[2]), fig.add_subplot(row_gs[3]), fig.add_subplot(row_gs[4])])

        for local_idx, k in enumerate(group):
            draw_population_mode_on_axis(
                ax=axes[local_idx],
                population_map=population_maps[k],
                height=height,
                width=width,
                k_scaler=k_scaler,
                title_text=f"k = {k}   Im(λ) ≈ {sorted_eigenvalues[k].imag:.4f}",
                radius=0.45,
                grid_linewidth=1.0,
                hide_spines=False,
                circle_edgecolor=None,
                circle_linewidth=0.0,
                title_fontsize=15,
                title_y=None,
                title_fontweight="bold",
            )

    out_name = f"{height}x{width}_dynamically_grouped_diagrams_final_with_correct_norm_RESTORED2.png"
    fig.savefig(out_name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    generate_relative_amplitude_colorbar(
        f"{height}x{width}_dynamically_grouped_diagrams_final.png",
        label="relative population amplitude ($p_k / max|p_k|$)",
    )


def main():
    # ---------------------------
    # System and visualization parameters
    # ---------------------------
    height = 10
    width = 10
    j = 1.0
    gamma = 0.1
    num_modes_to_visualize = 99

    tau = generate_grid_tau(height, width)
    liouvillian = build_liouvillian_dense(tau, j, gamma)
    lambdas, _, right = analyze_liouvillian_modes_dense_strict(liouvillian)
    plot_dynamically_grouped_diagrams(lambdas, right, height, width, num_modes_to_visualize)


if __name__ == "__main__":
    main()

