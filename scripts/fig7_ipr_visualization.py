"""Fig. 7 (v2, referee fix CF-2): mean IPR of the three slowest DISTINCT relaxation-rate
groups vs rewiring probability p (log axis), with run-to-run error bars (+-1 std).

Reads precalc/fig78_groups.npz produced by reduce_sweep_groups.py from the corrected
(rightmost-mode) sweep archive. Group IPR uses the basis-invariant diagonal weight of the
group's right-eigenspace (see reduce_sweep_groups.py); for a non-degenerate group it reduces
exactly to the legacy per-mode IPR (metrics.calculate_ipr, F-order reshape).
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
    p, ipr = z["p"], z["ipr"]            # ipr: (40, 30, 3)

    for gi, color in zip(range(3), LINE_COLORS):
        avg = np.nanmean(ipr[:, :, gi], axis=1)
        std = np.nanstd(ipr[:, :, gi], axis=1)

        fig, ax = plt.subplots(figsize=(WIDTH_THIRD, WIDTH_THIRD * 0.95), layout="constrained")
        ax.errorbar(p, avg, yerr=std, fmt="-o", capsize=2, markersize=2.5,
                    color=color, ecolor=color, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("$p$")
        ax.set_ylabel(rf"$\langle \mathrm{{IPR}}(k'={gi+1}) \rangle$")
        # compact exponent ticks every 2 decades (decimal labels collide at 42 mm width)
        ax.set_xticks([1e-4, 1e-2, 1e0])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(True, which="both", ls="--")
        ax.margins(x=0.05)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_pdf(fig, str(FIG_DIR / f"10x10_IPR_group_k_{gi+1}.pdf"))
    print("fig7 v2: 10x10_IPR_group_k_1..3 rebuilt")


if __name__ == "__main__":
    main()
