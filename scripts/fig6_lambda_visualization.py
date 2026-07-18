"""Fig. 6 (modes_log_scale): mean Re(lambda_k) vs rewiring probability p, and the
slow-mode separation Re(lambda_1 - lambda_2), on a log-p axis, with run-to-run error bars.

Note: Re(lambda_1 - lambda_2) is the separation between the two slowest modes, NOT the
conventional Liouvillian relaxation gap -Re(lambda_1) (which is panel (a), Re(lambda_1)).

Reads the precomputed spectral archive (rewiring_spectrum_data_*.npz). Only the
plotting is styled here; the data reduction (mean/std over runs) is unchanged.
"""
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.npz_io import parse_grouped_lambdas
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_COL, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"


def find_npz():
    """Newest rewiring_spectrum_data_*.npz found in precalc/ or the current dir."""
    cands = glob.glob(str(PRECALC / "rewiring_spectrum_data_*.npz")) + glob.glob("rewiring_spectrum_data_*.npz")
    if not cands:
        raise FileNotFoundError("no rewiring_spectrum_data_*.npz in scripts/precalc or cwd")
    return max(cands, key=os.path.getctime)


def _log_errorbar(p_vals, avg, std, ylabel, color, out_name):
    """One styled log-p error-bar panel at single-column width."""
    fig, ax = plt.subplots(figsize=(WIDTH_COL, WIDTH_COL * 0.78), layout="constrained")
    ax.errorbar(p_vals, avg, yerr=std, fmt="-o", capsize=2, color=color, ecolor=color, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="--")
    ax.margins(x=0.03)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / out_name))


def main():
    apply_style()
    grouped = parse_grouped_lambdas(find_npz())
    p_values = sorted(grouped.keys())
    p_pos = [p for p in p_values if p > 0]  # log axis needs strictly positive p

    # --- Re(lambda_k) for the three slowest non-trivial modes ---
    for k, color in zip((1, 2, 3), LINE_COLORS):
        avg = [np.mean([arr[k].real for arr in grouped[p]]) for p in p_pos]
        std = [np.std([arr[k].real for arr in grouped[p]]) for p in p_pos]
        _log_errorbar(p_pos, avg, std, rf"$\mathrm{{Re}}(\lambda_{k})$", color,
                      f"mode_{k}_p_log_ENG.pdf")

    # --- slow-mode separation Re(lambda_1) - Re(lambda_2) (not the Liouvillian gap -Re(lambda_1)) ---
    avg_gap = [np.mean([arr[1].real - arr[2].real for arr in grouped[p] if len(arr) > 2]) for p in p_pos]
    std_gap = [np.std([arr[1].real - arr[2].real for arr in grouped[p] if len(arr) > 2]) for p in p_pos]
    _log_errorbar(p_pos, avg_gap, std_gap, r"$\mathrm{Re}(\lambda_1-\lambda_2)$", LINE_COLORS[3],
                  "gap_p_log_ENG.pdf")


if __name__ == "__main__":
    main()
