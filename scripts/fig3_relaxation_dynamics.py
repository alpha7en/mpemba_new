from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt
import numpy as np

from qdyn_research import (
    QuantumSimulatorCore,
    build_liouvillian_dense,
    create_boundary_state,
    create_checkerboard_state,
    create_entangled_diagonal_state,
    create_four_corners_state,
    create_inner_corners_state,
    create_localized_state,
    create_mixed_diagonal_state,
    create_opposite_corners_state,
    create_top_bottom_edges_state,
    generate_grid_tau,
)


def find_intersection(t1, d1, t2, d2):
    for i in range(1, len(t1)):
        if (d1[i - 1] - d2[i - 1]) * (d1[i] - d2[i]) < 0:
            intersect_t = t1[i - 1] + (t1[i] - t1[i - 1]) * abs(d1[i - 1] - d2[i - 1]) / abs((d1[i] - d2[i]) - (d1[i - 1] - d2[i - 1]))
            intersect_d = d1[i - 1] + (d1[i] - d1[i - 1]) * (intersect_t - t1[i - 1]) / (t1[i] - t1[i - 1])
            return intersect_t, intersect_d
    return None


def main():
    simulator = QuantumSimulatorCore()

    # ---------------------------
    # Dynamics parameters
    # ---------------------------
    j_val = 1.0
    gamma_val = 0.5

    # ---------------------------
    # Initial-state ensemble
    # ---------------------------
    generators = {
        "1. Center (Mixed)": create_localized_state,
        "2. Opposite corners (Mixed)": create_opposite_corners_state,
        "3. Four corners (Mixed)": create_four_corners_state,
        "4. Mixed diagonal": create_mixed_diagonal_state,
        "5. Entangled diagonal": create_entangled_diagonal_state,
        "6. Inner corners (Mixed)": create_inner_corners_state,
        "7. Edges (top/bottom) (Mixed)": create_top_bottom_edges_state,
        "8. Checkerboard (Mixed)": create_checkerboard_state,
        "9. Boundary (Mixed)": create_boundary_state,
    }

    for n in range(10, 11):
        fig, ax = plt.subplots(figsize=(12, 8))
        tau = generate_grid_tau(n, n)

        # Lindblad equation in vectorized form: d/dt vec(rho)=L vec(rho), vec uses order='F'.
        liouvillian = build_liouvillian_dense(tau, j_val, gamma_val)

        case_data = {}
        for name, generator in generators.items():
            rho_initial, _ = generator(n, n)
            if rho_initial is None:
                continue
            trace = np.trace(rho_initial)
            if np.isclose(trace, 0):
                continue

            rho_initial = rho_initial / trace
            t, d, elapsed = simulator.run_simulation(liouvillian, rho_initial)
            print(f"{name}: {elapsed:.3f}s, {len(t)} points")

            if name.startswith("3. Four corners") or name.startswith("4. Mixed diagonal") or name.startswith("5. Entangled diagonal"):
                ax.plot(t, d, label=name, lw=3)
                case_data[name] = (t, d)
            else:
                ax.plot(t, d, label=name, lw=1, alpha=0.5)

        if "3. Four corners (Mixed)" in case_data and "5. Entangled diagonal" in case_data:
            t3, d3 = case_data["3. Four corners (Mixed)"]
            t5, d5 = case_data["5. Entangled diagonal"]
            inter = find_intersection(t3, d3, t5, d5)
            if inter:
                ax.plot(inter[0], inter[1], "o", markersize=8, color="red")

        if "5. Entangled diagonal" in case_data and "4. Mixed diagonal" in case_data:
            t5, d5 = case_data["5. Entangled diagonal"]
            t4, d4 = case_data["4. Mixed diagonal"]
            inter = find_intersection(t5, d5, t4, d4)
            if inter:
                ax.plot(inter[0], inter[1], "o", markersize=8, color="red")

        ax.set_title(f"Relaxation dynamics for {n}x{n} lattice", fontsize=16)
        ax.set_xlabel("Time, t")
        ax.set_ylabel("Distance metric D(t)")
        ax.grid(True, which="both", linestyle="--")
        ax.legend(loc="upper right", fontsize=10)

        fig.tight_layout()
        fig.savefig(f"9states_modes_simulation_line_{n}x{n}_with_selection_ENG.png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


if __name__ == "__main__":
    main()

