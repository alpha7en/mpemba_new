import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.npz_io import parse_grouped_lambdas


def main():
    # ---------------------------
    # Input dataset
    # ---------------------------
    files = glob.glob("*.npz")
    if not files:
        raise FileNotFoundError("No .npz files found in current directory")
    latest = max(files, key=os.path.getctime)
    grouped = parse_grouped_lambdas(latest)
    if not grouped:
        raise RuntimeError("No *_lambdas arrays found in file")

    p_values = sorted(grouped.keys())
    num_modes = len(next(iter(grouped.values()))[0])

    # ---------------------------
    # Output parameters
    # ---------------------------
    out_dir = "visualisation_plots_ENG_captions"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    p_values_log = [p for p in p_values if p > 0]
    p_idx_log = [p_values.index(p) for p in p_values_log]

    for mode_idx in range(num_modes):
        # For each p: mean and standard deviation of Re(lambda_k) across runs.
        avg_values = []
        std_values = []
        for p in p_values:
            samples = [arr[mode_idx].real for arr in grouped[p]]
            avg_values.append(np.mean(samples))
            std_values.append(np.std(samples))

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(p_values, avg_values, yerr=std_values, fmt="-o", capsize=5, color="darkblue", alpha=0.8)
        ax.set_title(f"Behavior of mode lambda_{mode_idx} (linear p)")
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel(f"Average Re(lambda_{mode_idx})")
        ax.grid(True, which="both", linestyle="--")
        if mode_idx == 0:
            ax.set_ylim(-0.1, 0.1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mode_{mode_idx}_p_linear_ENG.png"))
        plt.close(fig)

        if p_values_log:
            avg_log = [avg_values[i] for i in p_idx_log]
            std_log = [std_values[i] for i in p_idx_log]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(p_values_log, avg_log, yerr=std_log, fmt="-o", capsize=5, color="darkred", alpha=0.8)
            ax.set_title(f"Behavior of mode lambda_{mode_idx} (log p)")
            ax.set_xlabel("Rewiring probability p")
            ax.set_ylabel(f"Average Re(lambda_{mode_idx})")
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--")
            if mode_idx == 0:
                ax.set_ylim(-0.1, 0.1)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"mode_{mode_idx}_p_log_ENG.png"))
            plt.close(fig)

    if num_modes >= 3:
        # Spectral gap proxy: Re(lambda_1) - Re(lambda_2).
        avg_gap = []
        std_gap = []
        for p in p_values:
            gaps = [arr[1].real - arr[2].real for arr in grouped[p] if len(arr) > 2]
            if gaps:
                avg_gap.append(np.mean(gaps))
                std_gap.append(np.std(gaps))
            else:
                avg_gap.append(np.nan)
                std_gap.append(np.nan)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(p_values, avg_gap, yerr=std_gap, fmt="-o", capsize=5, color="darkgreen", alpha=0.8)
        ax.set_title("Gap: Re(lambda_1) - Re(lambda_2) vs p (linear p)")
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("Average gap")
        ax.grid(True, which="both", linestyle="--")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gap_p_linear_ENG.png"))
        plt.close(fig)

        if p_values_log:
            avg_gap_log = [avg_gap[i] for i in p_idx_log]
            std_gap_log = [std_gap[i] for i in p_idx_log]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(p_values_log, avg_gap_log, yerr=std_gap_log, fmt="-o", capsize=5, color="purple", alpha=0.8)
            ax.set_title("Gap: Re(lambda_1) - Re(lambda_2) vs p (log p)")
            ax.set_xlabel("Rewiring probability p")
            ax.set_ylabel("Average gap")
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "gap_p_log_ENG.png"))
            plt.close(fig)


if __name__ == "__main__":
    main()

