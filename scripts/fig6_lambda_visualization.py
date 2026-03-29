import glob
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.npz_io import parse_grouped_lambdas


def _size_to_points(size):
    if isinstance(size, (int, float)):
        return float(size)
    return mpl.font_manager.font_scalings.get(str(size), 1.0) * plt.rcParams["font.size"]


def _apply_bold_double_text(ax, title_size, label_size, tick_size):
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(tick_size)
        tick_label.set_fontweight("bold")


def _tight_x_to_last_point(ax, x_values):
    if not x_values:
        return
    if len(x_values) == 1:
        # Visual horizontal padding only, so a single marker is not clipped by the frame.
        ax.margins(x=0.02)
        return

    x_min, x_max = x_values[0], x_values[-1]
    pad_fraction = 0.02  # Visual horizontal padding only (2% on each side).

    if ax.get_xscale() == "log" and x_min > 0 and x_max > 0:
        # Visual horizontal padding only, computed in log-space to match linear visual spacing.
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_span = log_max - log_min

        if log_span == 0:
            ax.margins(x=pad_fraction)
        else:
            ax.set_xlim(
                10 ** (log_min - log_span * pad_fraction),
                10 ** (log_max + log_span * pad_fraction),
            )
    else:
        # Visual horizontal padding only, computed from linear span.
        span = x_max - x_min
        pad = span * pad_fraction if span != 0 else max(abs(x_max) * pad_fraction, 1e-12)
        ax.set_xlim(x_min - pad, x_max + pad)


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

    doubled_title_size = 2 * _size_to_points(plt.rcParams["axes.titlesize"])
    doubled_label_size = 2 * _size_to_points(plt.rcParams["axes.labelsize"])
    doubled_tick_size = 2 * _size_to_points(plt.rcParams["xtick.labelsize"])

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
        ax.set_xlabel("p")
        ax.set_ylabel(f"Average Re(λ_{mode_idx})")
        ax.grid(True, which="both", linestyle="--")
        if mode_idx == 0:
            ax.set_ylim(-0.1, 0.1)
        _tight_x_to_last_point(ax, p_values)
        _apply_bold_double_text(ax, doubled_title_size, doubled_label_size, doubled_tick_size)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mode_{mode_idx}_p_linear_ENG.png"))
        plt.close(fig)

        if p_values_log:
            avg_log = [avg_values[i] for i in p_idx_log]
            std_log = [std_values[i] for i in p_idx_log]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(p_values_log, avg_log, yerr=std_log, fmt="-o", capsize=5, color="darkred", alpha=0.8)
            ax.set_xlabel("p")
            ax.set_ylabel(f"Average Re(λ_{mode_idx})")
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--")
            if mode_idx == 0:
                ax.set_ylim(-0.1, 0.1)
            _tight_x_to_last_point(ax, p_values_log)
            _apply_bold_double_text(ax, doubled_title_size, doubled_label_size, doubled_tick_size)
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
        ax.set_xlabel("p")
        ax.set_ylabel("Average λ gap")
        ax.grid(True, which="both", linestyle="--")
        _tight_x_to_last_point(ax, p_values)
        _apply_bold_double_text(ax, doubled_title_size, doubled_label_size, doubled_tick_size)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gap_p_linear_ENG.png"))
        plt.close(fig)

        if p_values_log:
            avg_gap_log = [avg_gap[i] for i in p_idx_log]
            std_gap_log = [std_gap[i] for i in p_idx_log]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(p_values_log, avg_gap_log, yerr=std_gap_log, fmt="-o", capsize=5, color="purple", alpha=0.8)
            ax.set_xlabel("p")
            ax.set_ylabel("Average gap")
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--")
            _tight_x_to_last_point(ax, p_values_log)
            _apply_bold_double_text(ax, doubled_title_size, doubled_label_size, doubled_tick_size)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "gap_p_log_ENG.png"))
            plt.close(fig)


if __name__ == "__main__":
    main()

