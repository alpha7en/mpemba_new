import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.metrics import calculate_ipr
from qdyn_research.npz_io import parse_grouped_vectors


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
        ax.set_title(f"Localization of mode lambda_{mode_idx} (IPR)")
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel(f"<IPR(lambda_{mode_idx})>")
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{n // 10}x{n // 10} IPR graph mode k={mode_idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()

