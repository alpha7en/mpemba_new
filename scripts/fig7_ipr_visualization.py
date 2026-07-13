"""Fig. 7 (ipr_graphs): mean IPR of the three slowest non-trivial modes vs rewiring
probability p (log axis), with run-to-run error bars. Reads the precomputed
rewiring_spectrum_data_*.npz. Only the plotting is styled; the IPR computation is
unchanged (metrics.calculate_ipr with Fortran-order reshape).
"""
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.metrics import calculate_ipr
from qdyn_research.npz_io import parse_grouped_vectors
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_THIRD, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"


def find_npz():
    cands = glob.glob(str(PRECALC / "rewiring_spectrum_data_*.npz")) + glob.glob("rewiring_spectrum_data_*.npz")
    if not cands:
        raise FileNotFoundError("no rewiring_spectrum_data_*.npz in scripts/precalc or cwd")
    return max(cands, key=os.path.getctime)


def main():
    apply_style()
    grouped = parse_grouped_vectors(find_npz())
    p_values = sorted(grouped.keys())
    p_pos = [p for p in p_values if p > 0]

    n_sq = next(iter(grouped.values()))[0].shape[0]
    sites = int(np.sqrt(n_sq))       # density-matrix dimension (100 for 10x10)
    side = int(np.sqrt(sites))       # lattice side (10)

    for mode, color in zip((1, 2, 3), LINE_COLORS):
        avg, std = [], []
        for p in p_pos:
            iprs = [calculate_ipr(V[:, mode], sites, reshape_order="F") for V in grouped[p]]
            avg.append(np.mean(iprs))
            std.append(np.std(iprs))

        fig, ax = plt.subplots(figsize=(WIDTH_THIRD, WIDTH_THIRD * 0.95), layout="constrained")
        ax.errorbar(p_pos, avg, yerr=std, fmt="-o", capsize=2, color=color, ecolor=color, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("$p$")
        ax.set_ylabel(rf"$\langle \mathrm{{IPR}}(\lambda_{mode}) \rangle$")
        # compact exponent ticks every 2 decades (decimal labels collide at 42 mm width)
        ax.set_xticks([1e-4, 1e-2, 1e0])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(True, which="both", ls="--")
        ax.margins(x=0.05)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_pdf(fig, str(FIG_DIR / f"{side}x{side}_IPR_graph_mode_k_{mode}.pdf"))


if __name__ == "__main__":
    main()
