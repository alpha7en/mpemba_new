import glob
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.metrics import calculate_ipr
from qdyn_research.npz_io import parse_grouped_vectors


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
    # Input dataset
    # ---------------------------
    files = glob.glob("rewiring_spectrum_data_*.npz")
    if not files:
        raise FileNotFoundError("No rewiring_spectrum_data_*.npz files in current directory")
    latest = max(files, key=os.path.getctime)

    grouped = parse_grouped_vectors(latest)
    if not grouped:
        raise RuntimeError("No grouped vectors found in file")

    p_values = sorted(grouped.keys())
    first_set = next(iter(grouped.values()))[0]
    n_sq, num_modes = first_set.shape
    n = int(np.sqrt(n_sq))

    modes = [1, 2, 3]
    if num_modes <= max(modes):
        raise RuntimeError(f"Need at least {max(modes) + 1} modes, found {num_modes}")

    # ---------------------------
    # Output parameters
    # ---------------------------
    out_dir = "IPR from p images ENG"
    os.makedirs(out_dir, exist_ok=True)

    for mode_idx in modes:
        # IPR = sum(|p_i|^4) / (sum(|p_i|^2))^2 from diagonal mode populations p_i.
        avg = []
        std = []
        for p in p_values:
            iprs = [calculate_ipr(run_vectors[:, mode_idx], n, reshape_order="F") for run_vectors in grouped[p]]
            avg.append(np.mean(iprs))
            std.append(np.std(iprs))

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(p_values, avg, yerr=std, fmt="-o", capsize=5, alpha=0.9)

        doubled_label_size = 2 * _size_to_points(plt.rcParams["axes.labelsize"])
        doubled_tick_size = 2 * _size_to_points(plt.rcParams["xtick.labelsize"])

        ax.set_xlabel("p")
        ax.set_ylabel(f"<IPR(λ_{mode_idx})>")
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--")

        _add_log_x_padding(ax, p_values, pad_fraction=0.02)
        _apply_bold_double_text(ax, doubled_label_size, doubled_tick_size)

        # Layout spacing only: slightly reduce bottom room while keeping x-label fully visible.
        fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.97])

        fig.savefig(os.path.join(out_dir, f"{n // 10}x{n // 10} IPR graph mode k={mode_idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()

