from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from qdyn_research import generate_rewired_grid_tau, run_average_path_experiment


def _size_to_points(size):
    if isinstance(size, (int, float)):
        return float(size)
    return mpl.font_manager.font_scalings.get(str(size), 1.0) * plt.rcParams["font.size"]


def _apply_bold_double_text(ax, label_size, tick_size):
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(tick_size)
        tick_label.set_fontweight("bold")


def _add_log_x_padding(ax, x_values, pad_fraction=0.02):
    positive_x = [x for x in x_values if x > 0]
    if not positive_x:
        return
    if len(positive_x) == 1:
        # Visual horizontal padding only, so the marker is not clipped by the frame.
        ax.margins(x=pad_fraction)
        return

    # Visual horizontal padding only, computed in log-space for consistent appearance.
    log_min = np.log10(positive_x[0])
    log_max = np.log10(positive_x[-1])
    log_span = log_max - log_min
    if log_span == 0:
        ax.margins(x=pad_fraction)
    else:
        ax.set_xlim(
            10 ** (log_min - log_span * pad_fraction),
            10 ** (log_max + log_span * pad_fraction),
        )


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

    doubled_label_size = 2 * _size_to_points(plt.rcParams["axes.labelsize"])
    doubled_tick_size = 2 * _size_to_points(plt.rcParams["xtick.labelsize"])

    ax.set_xlabel("p")
    ax.set_ylabel("Average shortest path <L>")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--")

    _add_log_x_padding(ax, p_vals, pad_fraction=0.02)
    _apply_bold_double_text(ax, doubled_label_size, doubled_tick_size)

    # Layout spacing only: slightly reduce bottom room while keeping x-label fully visible.
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.97])

    fig.savefig(f"log{height}x{width}_p_L.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

