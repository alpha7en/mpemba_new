from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt
import numpy as np

from qdyn_research import generate_rewired_grid_tau, run_average_path_experiment


def main():
    # ---------------------------
    # Grid parameters
    # ---------------------------
    height = 10
    width = 10

    # ---------------------------
    # Ensemble parameters
    # ---------------------------
    runs_per_p = 1000

    # Sweep in p for the topological observable <L>, the mean shortest-path length.
    log_p_values = np.logspace(-4, 0, num=100)
    p_values = np.concatenate(([0.0], log_p_values))

    results = run_average_path_experiment(
        height=height,
        width=width,
        n_runs=runs_per_p,
        p_values=p_values,
        generator=generate_rewired_grid_tau,
    )

    p_vals = list(results.keys())
    l_vals = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_vals, l_vals, marker="o", linestyle="-")
    ax.set_xlabel("Rewiring probability p")
    ax.set_ylabel("Average shortest path <L>")
    ax.set_title("<L>(p) for rewired grid")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--")
    fig.savefig(f"log{height}x{width}_p_L.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

