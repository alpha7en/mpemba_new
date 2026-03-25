import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import (
    analyze_liouvillian_modes_dense_strict,
    build_liouvillian_dense,
    calculate_distance_metric_logm,
    create_boundary_state,
    create_checkerboard_state,
    create_entangled_diagonal_state,
    create_four_corners_state,
    create_inner_corners_state,
    create_localized_state,
    create_mixed_diagonal_state,
    create_opposite_corners_state,
    create_top_bottom_edges_state,
    draw_population_mode_on_axis,
    generate_grid_tau,
    project_rho_on_modes,
)


def analyze_projections(coefficients: np.ndarray):
    contributions = np.abs(coefficients[1:]) ** 2
    total = np.sum(contributions)
    if total < 1e-12:
        return np.zeros_like(contributions), 8, np.array([], dtype=int)
    normalized = contributions / total
    mode_indices = np.arange(1, len(normalized) + 1)
    sorted_modes = mode_indices[np.argsort(normalized)[::-1]]
    top_3 = sorted_modes[:3]
    top_8 = sorted_modes[:8]
    k_max = int(np.max(top_8)) if len(top_8) > 0 else 8
    return normalized, k_max, top_3


def visualize_experiment_for_n(n, j, gamma, state_generators):
    n_sites = n * n
    tau = generate_grid_tau(n, n)
    liouvillian = build_liouvillian_dense(tau, j, gamma)
    _, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)

    # Initial states are ranked by D(rho)=log(N)+Tr(rho log rho), i.e. distance to equilibrium.
    results = []
    for name, generator in state_generators.items():
        rho, indices = generator(n, n)
        if rho is None:
            continue
        metric = calculate_distance_metric_logm(rho)
        results.append({"name": name, "rho": rho, "indices": indices, "metric": metric})
    results.sort(key=lambda x: x["metric"], reverse=True)

    global_max_contrib = 0.0
    max_abs_mode_pop = 0.0

    for item in results:
        coeffs = project_rho_on_modes(item["rho"], left_vecs, right_vecs, order="F")
        contribs, k_max, top_3 = analyze_projections(coeffs)
        item.update({"contribs": contribs, "k_max": k_max, "top_3": top_3})

        if len(contribs) > 0:
            global_max_contrib = max(global_max_contrib, contribs.max())

        for k in top_3:
            rho_k = right_vecs[:, k].reshape((n_sites, n_sites), order="F")
            pop_k = np.diag(rho_k).real
            max_abs_mode_pop = max(max_abs_mode_pop, np.max(np.abs(pop_k)))

    k_scaler = 1.0 / (max_abs_mode_pop + 1e-9)

    fig = plt.figure(figsize=(12, 5 * len(results)))
    outer_gs = gridspec.GridSpec(len(results), 2, figure=fig, width_ratios=[1, 1.5], wspace=0.3)

    for i, item in enumerate(results):
        ax_rho = fig.add_subplot(outer_gs[i, 0])
        ax_rho.set_title(item["name"], fontsize=11)
        ax_rho.set_xticks(np.arange(n))
        ax_rho.set_yticks(np.arange(n))
        ax_rho.set_xticklabels([])
        ax_rho.set_yticklabels([])
        schematic = np.zeros((n, n))
        val = 1.0 / len(item["indices"]) if item["indices"] else 0
        for idx in item["indices"]:
            row, col = divmod(idx, n)
            schematic[row, col] = val
        ax_rho.imshow(schematic, cmap="viridis", vmin=0, vmax=max(0.001, schematic.max()), aspect="equal")

        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i, 1], height_ratios=[1, 1], hspace=0.6)
        bar_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_gs[0], height_ratios=[0.15, 1], hspace=0.3)

        ax_text = fig.add_subplot(bar_gs[0])
        ax_text.axis("off")
        ax_text.text(0.5, 0.5, f"Метрика D(ρ(0)) = {item['metric']:.4f}", ha="center", va="center", fontsize=12)

        ax_bar = fig.add_subplot(bar_gs[1])
        ax_bar.set_yscale("log")
        ymin = 0.01
        ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))

        contribs = item["contribs"]
        local_k_max = item["k_max"]
        modes_to_plot = np.arange(1, local_k_max + 1)
        contribs_percent = contribs[:local_k_max] * 100
        valid_idx = np.where(contribs_percent > ymin)[0]

        if local_k_max <= 40:
            if len(valid_idx) > 0:
                ax_bar.bar(modes_to_plot[valid_idx], contribs_percent[valid_idx], color="skyblue", width=0.8)
            ax_bar.set_xticks(modes_to_plot)
            ax_bar.set_xticklabels([f"{k}" for k in modes_to_plot], rotation=90, fontsize=8)
        else:
            if len(valid_idx) > 0:
                ax_bar.vlines(modes_to_plot[valid_idx], ymin, contribs_percent[valid_idx], color="skyblue")
            tick_step = int(np.ceil(local_k_max / 20))
            ticks = np.arange(1, local_k_max + 1, tick_step)
            ax_bar.set_xticks(ticks)
            ax_bar.set_xticklabels([f"{k}" for k in ticks], rotation=90, fontsize=8)

        ax_bar.set_xlabel("Индекс моды k (от 1)")
        ax_bar.set_ylabel("Относительный вклад, % (log)")
        ax_bar.set_xlim(0.5, local_k_max + 0.5)
        ymax = global_max_contrib * 100 * 1.15 if global_max_contrib > 0 else 1
        ax_bar.set_ylim(bottom=ymin, top=max(ymax, ymin * 10))

        bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=right_gs[1], wspace=0.1)
        top_indices = item["top_3"]
        for j, k in enumerate(top_indices):
            ax_mode = fig.add_subplot(bottom_gs[j])
            rho_k = right_vecs[:, k].reshape((n_sites, n_sites), order="F")
            pop_map = np.diag(rho_k).real.reshape((n, n))
            contrib_percent = item["contribs"][k - 1] * 100
            draw_population_mode_on_axis(
                ax_mode,
                pop_map,
                n,
                n,
                k_scaler,
                f"Mode k={k}\n({contrib_percent:.1f}%)",
            )
        for j in range(len(top_indices), 3):
            fig.add_subplot(bottom_gs[j]).axis("off")

    fig.tight_layout()
    fig.savefig(f"9states_modes_bar_chart_with_modes_{n}x{n}_log_afterdebag_gamma01.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    # ---------------------------
    # Dynamics parameters
    # ---------------------------
    j_val = 1.0
    gamma_val = 0.1

    # ---------------------------
    # Initial-state ensemble
    # ---------------------------
    state_generators = {
        "1. Центр (Смесь)": create_localized_state,
        "2. Противоп. углы (Смесь)": create_opposite_corners_state,
        "3. Четыре угла (Смесь)": create_four_corners_state,
        "4. Смешанная диагональ": create_mixed_diagonal_state,
        "5. Запутанная диагональ": create_entangled_diagonal_state,
        "6. Внутр. углы (Смесь)": create_inner_corners_state,
        "7. Края (верх/низ) (Смесь)": create_top_bottom_edges_state,
        "8. Шахматка (Смесь)": create_checkerboard_state,
        "9. Граница (Смесь)": create_boundary_state,
    }

    for n in range(10, 11):
        visualize_experiment_for_n(n, j_val, gamma_val, state_generators)


if __name__ == "__main__":
    main()

