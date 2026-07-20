"""Fig. 8 (v2, referee fix CF-2): mean maximum community-overlap of the three slowest
DISTINCT relaxation-rate groups vs rewiring probability p (log axis), +-1 std error bars.

Reads precalc/fig78_groups.npz produced by reduce_sweep_groups.py from the corrected
(rightmost-mode) sweep archive. The overlap mass is the basis-invariant diagonal weight of
the group's right-eigenspace; Louvain partitions are seeded (random_state=42) and computed
once per graph (see reduce_sweep_groups.py). For a non-degenerate group this reduces exactly
to the legacy per-mode overlap (mass |(rho_k)_ii|^2, F-order reshape).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_THIRD, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig78_groups.npz"


def main():
    apply_style()
    z = np.load(CACHE)
    p, overlap = z["p"], z["overlap"]        # overlap: (40, 30, 3)

    for gi, color in zip(range(3), LINE_COLORS):
        avg = np.nanmean(overlap[:, :, gi], axis=1)
        std = np.nanstd(overlap[:, :, gi], axis=1)

        fig, ax = plt.subplots(figsize=(WIDTH_THIRD, WIDTH_THIRD * 0.95), layout="constrained")
        ax.errorbar(p, avg, yerr=std, fmt="-o", capsize=2, markersize=2.5,
                    color=color, ecolor=color, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("$p$")
        ax.set_ylabel(rf"max overlap ($k'={gi+1}$)")
        # compact exponent ticks every 2 decades (decimal labels collide at 42 mm width)
        ax.set_xticks([1e-4, 1e-2, 1e0])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(True, which="both", ls="--")
        ax.margins(x=0.05)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_pdf(fig, str(FIG_DIR / f"10x10_overlap_group_k_{gi+1}.pdf"))
    print("fig8 v2: 10x10_overlap_group_k_1..3 rebuilt")


if __name__ == "__main__":
    main()
